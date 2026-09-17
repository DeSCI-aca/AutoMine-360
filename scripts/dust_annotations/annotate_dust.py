#!/usr/bin/env python3
"""Local six-camera dust annotator. Python 3; Pillow optional for thumbnails."""
import argparse, csv, io, json, os, tempfile, threading, webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

LEVELS = {0: '无尘', 1: '轻度', 2: '中度', 3: '重度'}

def atomic_write(path, text):
    fd, temp = tempfile.mkstemp(prefix='.'+path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(temp, str(path))
    finally:
        if os.path.exists(temp): os.unlink(temp)

class Store:
    def __init__(self, root, output):
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.scenes = {}
        for scene in sorted(self.root.glob('scene_*')):
            cams = {}
            for c in range(6):
                directory = scene/'mav0'/('cam%d'%c)/'data'
                cams[str(c)] = {f.stem: f for f in directory.glob('*') if f.suffix.lower() in ('.png','.jpg','.jpeg')}
            stamps = set().union(*(set(x) for x in cams.values()))
            if not stamps: continue
            stamps = sorted(stamps, key=lambda x: (float(x),x))
            path = self.output/(scene.name+'.json')
            data = json.loads(path.read_text()) if path.exists() else dict(schema_version=1, scene=scene.name, root=str(self.root), current_index=0, frames={})
            if data.get('root') != str(self.root): raise ValueError('标注目录属于另一数据根目录，请换 --output')
            old = data.get('current_timestamp')
            data['current_index'] = stamps.index(old) if old in stamps else min(data.get('current_index',0),len(stamps)-1)
            data.setdefault('forward_overrides', {})
            self.scenes[scene.name] = dict(cams=cams, stamps=stamps, data=data)
        if not self.scenes: raise ValueError('没有找到 scene_*/mav0/cam*/data 图像')
        self.lock = threading.RLock()

    def visit(self, name, index, inherit=None, overrides=None):
        s=self.scenes[name]; index=max(0,min(index,len(s['stamps'])-1)); stamp=s['stamps'][index]
        frames=s['data']['frames']
        if stamp not in frames:
            # New frames inherit only from the preceding frame, never from future frames.
            previous=inherit if inherit is not None else (frames.get(s['stamps'][index-1],{}) if index else {})
            frames[stamp]={}
            for c in range(6):
                key=str(c); path=s['cams'][key].get(stamp)
                if not path: continue
                prior=previous.get(key, {})
                frames[stamp][key]=dict(cam='cam%d'%c, name=path.name, level=prior.get('level',0),
                                       source='inherited' if prior else 'default',
                                       inherited_from=s['stamps'][index-1] if prior and index else None)
        # A fresh manual choice overrides this camera even in previously visited frames.
        # Apply on Next only; untouched future frames remain unchanged until visited.
        for key, level in (overrides or {}).items():
            if key in frames[stamp]:
                frames[stamp][key].update(level=level, source='inherited',
                                          inherited_from=s['stamps'][index-1] if index else None)
        s['data']['current_index']=index;s['data']['current_timestamp']=stamp
        self.save(name)
        return self.snapshot(name)

    def save(self, name):
        atomic_write(self.output/(name+'.json'),json.dumps(self.scenes[name]['data'],ensure_ascii=False,indent=2))

    def snapshot(self,name):
        s=self.scenes[name];d=s['data'];i=d['current_index'];stamp=s['stamps'][i];records=d['frames'].get(stamp,{})
        return dict(scene=name,scenes=list(self.scenes),index=i,total=len(s['stamps']),timestamp=stamp,
                    visited=len(d['frames']),output=str(self.output),cams=[dict(cam='cam%d'%c,
                    name=s['cams'][str(c)][stamp].name if stamp in s['cams'][str(c)] else None,
                    level=records.get(str(c),{}).get('level',0),source=records.get(str(c),{}).get('source','missing')) for c in range(6)])

    def action(self, request):
        with self.lock:
            name=request.get('scene',next(iter(self.scenes)));s=self.scenes[name];kind=request.get('action','open')
            if kind=='open':return self.visit(name,s['data']['current_index'])
            if request.get('index') != s['data']['current_index']:raise ValueError('页面已过期，请刷新；请勿同时在多个标签页标注')
            i=s['data']['current_index'];stamp=s['stamps'][i]
            if kind=='label':
                c=str(int(request['cam']));level=int(request['level'])
                if level not in LEVELS or c not in s['data']['frames'][stamp]:raise ValueError('无效等级或该相机缺帧')
                s['data']['frames'][stamp][c].update(level=level,source='manual',inherited_from=None)
                s['data']['forward_overrides'][c]=level
                self.save(name);return self.snapshot(name)
            if kind=='next':
                if i+1>=len(s['stamps']):return self.snapshot(name)
                return self.visit(name,i+1,s['data']['frames'][stamp],s['data']['forward_overrides'])
            if kind=='prev':
                if i==0:return self.snapshot(name)
                # Backward browsing must not propagate a later label into earlier frames.
                # Select a level again to start a new forward correction pass.
                s['data']['forward_overrides']={}
                return self.visit(name,i-1)
            if kind=='export':
                self.export();return self.snapshot(name)
            raise ValueError('Unknown action')

    def export(self):
        columns=['scene','timestamp','cam','name','level','severity','source','inherited_from']
        allrows=[]
        for name,s in self.scenes.items():
            for stamp,records in sorted(s['data']['frames'].items(),key=lambda kv:float(kv[0])):
                for c,r in sorted(records.items()):
                    allrows.append(dict(scene=name,timestamp=stamp,cam=r['cam'],name=r['name'],level=r['level'],
                                        severity=LEVELS[r['level']],source=r['source'],inherited_from=r.get('inherited_from') or ''))
        for filename, rows in [('annotations.csv',allrows),('dust_only.csv',[r for r in allrows if r['level']>0])]:
            f=io.StringIO();w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
            atomic_write(self.output/filename,f.getvalue())

@lru_cache(maxsize=48)
def thumbnail(path,mtime):
    try:
        from PIL import Image,ImageOps
    except ImportError:
        return Path(path).read_bytes(), 'image/png' if Path(path).suffix.lower()=='.png' else 'image/jpeg'
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((800,600))
        b=io.BytesIO();im.save(b,'JPEG',quality=85);return b.getvalue(),'image/jpeg'

HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>六路扬尘标注</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#181b21;color:#eee;font:15px system-ui,sans-serif}header{padding:10px 16px;background:#252a33;display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,select{font:inherit;padding:8px 14px;border:0;border-radius:5px;cursor:pointer}button:disabled{opacity:.4;cursor:default}#grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;padding:10px}figure{margin:0;border:5px solid transparent;background:#252a33;border-radius:6px;cursor:pointer;min-width:0}figure img{display:block;width:100%;height:calc((100vh - 225px)/2);min-height:150px;object-fit:contain}figcaption{padding:5px 8px;font-size:13px;overflow-wrap:anywhere}figure[data-level="1"]{border-color:#f4d35e}figure[data-level="2"]{border-color:#ff922b}figure[data-level="3"]{border-color:#f03e3e}.missing{opacity:.5;cursor:default}.hint{padding:6px 16px;color:#cbd0d9}.badge{padding:2px 7px;border-radius:3px;background:#12151a}dialog{background:#303642;color:white;border:1px solid #667;border-radius:10px;max-width:650px}dialog::backdrop{background:#0009}#levels button{margin:8px;padding:18px;font-weight:bold}.error{color:#ff9999}#modalimg{max-width:100%;max-height:45vh}#status{font-size:13px}@media(max-width:800px){#grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
<header><b>六路扬尘标注</b><select id="scene"></select><button id="prev">← 上一帧</button><button id="next">下一帧 →</button><span id="position"></span><button id="export">导出 CSV</button></header>
<div class="hint">点击图片选择等级：0 无尘（清除） · <span style="color:#f4d35e">1 轻度</span> · <span style="color:#ff922b">2 中度</span> · <span style="color:#f03e3e">3 重度</span>。重新选择等级后，下一帧持续继承并覆盖该相机的旧记录（含0无尘）；仅回退浏览不覆盖。未浏览帧不导出。</div>
<div class="hint" id="status">正在加载…</div><div id="grid"></div>
<dialog id="dialog"><h3 id="title"></h3><img id="modalimg"><p>选择即确认并自动保存。无尘请选择 0，停止该相机的扬尘状态延续。</p><div id="levels"><button data-v="0">0 无尘</button><button data-v="1" style="background:#f4d35e">1 轻度</button><button data-v="2" style="background:#ff922b">2 中度</button><button data-v="3" style="background:#f03e3e">3 重度</button></div><button id="cancel">取消（保持原状态）</button></dialog>
<script>
let state=null,busy=false,selected=null;const names=['无尘','轻度','中度','重度'];const $=id=>document.getElementById(id);
function controls(){ $('prev').disabled=busy||!state||state.index===0;$('next').disabled=busy||!state||state.index===state.total-1;$('scene').disabled=busy;$('export').disabled=busy;document.querySelectorAll('#levels button').forEach(b=>b.disabled=busy);}
async function api(action,extra={}){if(busy)return;busy=true;controls();try{let r=await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,scene:state?.scene,index:state?.index,...extra})});let d=await r.json();if(!r.ok)throw Error(d.error);state=d;render();if(action==='export')$('status').textContent='已导出 annotations.csv 和 dust_only.csv → '+state.output;return true}catch(e){$('status').textContent='保存/加载失败：'+e.message;$('status').className='error';return false}finally{busy=false;controls()}}
function render(){let old=$('scene').value;$('scene').replaceChildren(...state.scenes.map(s=>{let o=document.createElement('option');o.value=s;o.textContent=s;return o}));$('scene').value=state.scene;$('position').textContent=`第 ${state.index+1} / ${state.total} 帧 · ${state.timestamp}`;$('status').className='';$('status').textContent=`自动保存成功 · 已访问 ${state.visited} 帧 · ${state.output}`;let grid=$('grid');grid.replaceChildren();state.cams.forEach((c,i)=>{let f=document.createElement('figure');f.dataset.level=c.name?c.level:0;let im=document.createElement('img');if(c.name){im.src=`/image?scene=${encodeURIComponent(state.scene)}&index=${state.index}&cam=${i}`;im.alt=c.cam;im.onerror=()=>{im.alt='图像读取失败：'+c.name;f.classList.add('error')};f.onclick=()=>{if(busy)return;selected=i;$('title').textContent=c.cam+' · '+c.name+' · 当前：'+names[c.level];$('modalimg').src=im.src;$('dialog').showModal()}}else{f.classList.add('missing');im.alt='该时间戳缺少 '+c.cam+' 图像，不记录标签'}let cap=document.createElement('figcaption');cap.textContent=c.cam+' | '+(c.name||'缺帧')+' | '+(c.name?names[c.level]:'不标注')+(c.source==='inherited'?'（继承）':'');f.append(im,cap);grid.append(f)})}
$('prev').onclick=()=>api('prev');$('next').onclick=()=>api('next');$('export').onclick=()=>api('export');$('scene').onchange=()=>api('open',{scene:$('scene').value});$('cancel').onclick=()=>$('dialog').close();
async function label(v){if(selected===null||busy)return;if(await api('label',{cam:selected,level:v}))$('dialog').close()}
document.querySelectorAll('#levels button').forEach(b=>b.onclick=()=>label(Number(b.dataset.v)));
document.addEventListener('keydown',e=>{if(busy)return;if($('dialog').open){if(['0','1','2','3'].includes(e.key)){e.preventDefault();label(Number(e.key))}return}if(e.target.tagName==='SELECT')return;if(e.key==='ArrowRight'){e.preventDefault();if(state&&state.index<state.total-1)api('next')}if(e.key==='ArrowLeft'){e.preventDefault();if(state&&state.index>0)api('prev')}});api('open');
</script></html>'''

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('/home/ivan/project/IJRR'))
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'dust_annotations')
    ap.add_argument('--port',type=int,default=8765)
    ap.add_argument('--no-browser',action='store_true')
    args=ap.parse_args();store=Store(args.root,args.output)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,data,kind='application/json; charset=utf-8',status=200):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                url=urlparse(self.path)
                if url.path=='/':return self.reply(HTML.encode(),'text/html; charset=utf-8')
                if url.path=='/image':
                    q=parse_qs(url.query);s=store.scenes[q['scene'][0]];i=int(q['index'][0]);c=str(int(q['cam'][0]))
                    if not 0<=i<len(s['stamps']):raise ValueError('Bad index')
                    p=s['cams'][c][s['stamps'][i]];data,kind=thumbnail(str(p),p.stat().st_mtime_ns);return self.reply(data,kind)
                self.reply({'error':'Not found'},status=404)
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as e:self.reply({'error':str(e)},status=400)
        def do_POST(self):
            try:
                if self.path!='/api':return self.reply({'error':'Not found'},status=404)
                # Restrict browser writes to this local application's origin.
                origin=self.headers.get('Origin')
                if origin and origin!='http://'+self.headers.get('Host',''):return self.reply({'error':'Origin rejected'},status=403)
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<8192:raise ValueError('Invalid body')
                self.reply(store.action(json.loads(self.rfile.read(n))))
            except Exception as e:self.reply({'error':str(e)},status=400)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    url='http://127.0.0.1:%d/'%server.server_port
    print('打开 '+url+'\n标注保存到 '+str(store.output)+'\nCtrl+C 停止；重新运行可继续。',flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        with store.lock:store.export()
        server.server_close()

if __name__=='__main__':main()
