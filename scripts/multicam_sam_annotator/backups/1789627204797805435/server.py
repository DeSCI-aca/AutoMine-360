#!/usr/bin/env python3
import argparse,json,traceback
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from http.server import HTTPServer,BaseHTTPRequestHandler
from core import Project,Segmenter,encode,png,CLASSES
import cv2
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output');p.add_argument('--port',type=int,default=8770);p.add_argument('--backend',choices=['sam','sam2'],default='sam');p.add_argument('--checkpoint',default=str(Path(__file__).parent/'models/sam_vit_b_01ec64.pth'));p.add_argument('--sam2-config',default='configs/sam2.1/sam2.1_hiera_t.yaml');a=p.parse_args();root=Path(__file__).parent;project=Project(a.scene,a.output or root/'annotations'/Path(a.scene).name,Segmenter(a.checkpoint,a.backend,a.sam2_config));cv2.setNumThreads(2)
class Handler(BaseHTTPRequestHandler):
 def send(self,data,kind='application/json',code=200):
  if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
  self.send_response(code);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
 def do_GET(self):
  try:
   u=urlparse(self.path);q=parse_qs(u.query)
   if u.path=='/':self.send((root/'index.html').read_bytes(),'text/html')
   elif u.path=='/frame':self.send(project.data(int(q.get('f',['0'])[0])))
   elif u.path=='/image':self.send(cv2.imencode('.jpg',project.image(int(q['f'][0]),int(q['c'][0])))[1].tobytes(),'image/jpeg')
   else:self.send({'error':'not found'},code=404)
  except Exception as e:self.send({'error':str(e)},code=400)
 def do_POST(self):
  try:
   size=int(self.headers.get('Content-Length',0))
   if size>8_000_000:raise ValueError('request too large')
   d=json.loads(self.rfile.read(size));f=int(d.get('frame',0));c=int(d.get('cam',0));log=[]
   if self.path=='/sam':
    m,score=project.seg.predict(project.image(f,c),d.get('box'),d.get('points'),d.get('labels'));self.send(dict(mask=encode(m),score=score));return
   elif self.path=='/save':
    obj=project.add(d)
    if d.get('cross'):log=project.cross(f,c,obj)
   elif self.path=='/track':
    log=project.temporal(f)
    if d.get('cross'):
     sources=[(cam,list(project.entries(f,cam))) for cam in range(6)]
     for cam,objects in sources:
      for obj in objects:log+=project.cross(f,cam,obj)
   elif self.path=='/cross':log=project.cross(f,c,str(d['object']))
   elif self.path=='/stop':project.stop(f,str(d['object']))
   elif self.path=='/approve':
    for cam in range(6):
     for e in project.entries(f,cam).values():e['reviewed']=True
    project.save()
   elif self.path=='/export':log=project.export(f)
   else:raise ValueError('unknown action')
   self.send(dict(data=project.data(f),log=log))
  except Exception as e:traceback.print_exc();self.send(dict(error=str(e)),code=400)
print(f'http://127.0.0.1:{a.port}  scene={a.scene}',flush=True);HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
