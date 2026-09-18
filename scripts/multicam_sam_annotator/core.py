import ast,json,time,hashlib,io,base64,re
from pathlib import Path
import cv2,numpy as np,yaml
W,H=1024,768
CLASSES={0:'背景',1:'道路',2:'车辆',3:'行人',4:'设施',5:'地形',6:'天空'}
def atomic(p,b):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(b);tmp.replace(p)
def png(a):return cv2.imencode('.png',a)[1].tobytes()
def encode(a):return 'data:image/png;base64,'+base64.b64encode(png(a)).decode()
def decode(s):
 b=base64.b64decode(s.split(',',1)[1],validate=True);a=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_GRAYSCALE)
 if a is None or a.shape!=(H,W):raise ValueError('mask must be 1024x768')
 return (a>127).astype('uint8')*255
class Segmenter:
 def __init__(self,checkpoint,backend='sam',config=None):self.checkpoint=checkpoint;self.backend=backend;self.config=config;self.predictor=None;self.cached=None
 def predict(self,image,box,points=None,labels=None):
  import torch
  torch.set_num_threads(4)
  if self.predictor is None:
   if self.backend=='sam':
    from segment_anything import SamPredictor,sam_model_registry
    model=sam_model_registry['vit_b'](checkpoint=str(self.checkpoint));model.to('cuda' if torch.cuda.is_available() else 'cpu');self.predictor=SamPredictor(model)
   else:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    self.predictor=SAM2ImagePredictor(build_sam2(self.config,str(self.checkpoint),device='cuda' if torch.cuda.is_available() else 'cpu'))
  key=hashlib.sha256(image.tobytes()).digest()
  with torch.inference_mode():
   if key!=self.cached:self.predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));self.cached=key
   masks,scores,_=self.predictor.predict(box=np.array(box,dtype=np.float32) if box is not None else None,point_coords=np.array(points,dtype=np.float32) if points else None,point_labels=np.array(labels,dtype=np.int32) if points else None,multimask_output=True)
  return masks[int(np.argmax(scores))].astype('uint8')*255,float(np.max(scores))
def smart_segment(segmenter,image,box):
 box=np.asarray(box,dtype=float)
 if box.shape!=(4,) or not np.isfinite(box).all():raise ValueError('invalid box')
 x0,x1=sorted(np.clip(box[[0,2]],0,W-1));y0,y1=sorted(np.clip(box[[1,3]],0,H-1))
 if x1-x0<5 or y1-y0<5:raise ValueError('框太小，请重新框选')
 prompt=[float(x0),float(y0),float(x1),float(y1)]
 mask,score=segmenter.predict(image,prompt)
 # Select the connected component with strongest overlap with the user's box.
 n,components,stats,_=cv2.connectedComponentsWithStats((mask>0).astype('uint8'))
 if n<=1:raise ValueError('SAM未找到目标，请扩大框或手动画区域')
 roi=components[int(y0):int(y1)+1,int(x0):int(x1)+1];counts=np.bincount(roi.ravel(),minlength=n);counts[0]=0;k=int(counts.argmax())
 if counts[k]<20:raise ValueError('SAM结果与框选区域不一致')
 mask=(components==k).astype('uint8')*255;ys,xs=np.where(mask>0)
 refined=[max(0,int(xs.min())-3),max(0,int(ys.min())-3),min(W-1,int(xs.max())+3),min(H-1,int(ys.max())+3)]
 second,second_score=segmenter.predict(image,refined)
 union=np.count_nonzero((second>0)|(mask>0));intersection=np.count_nonzero((second>0)&(mask>0))
 if intersection/max(1,union)>.65 and second_score>=score-.05:mask=second;score=second_score
 ys,xs=np.where(mask>0);tight=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]
 return mask,score,tight
class Project:
 def __init__(self,scene,out,segmenter):
  self.scene=Path(scene).resolve();self.out=Path(out).resolve();self.out.mkdir(parents=True,exist_ok=True);self.seg=segmenter;self.images={};self.frames=sorted({p.stem for c in range(6) for p in (self.scene/'mav0'/f'cam{c}'/'data').glob('*.png')},key=float)
  if not self.frames:raise ValueError('No timestamp PNG images under scene/mav0/cam*/data')
  f=self.out/'project.json';self.state=json.loads(f.read_text()) if f.exists() else dict(scene=str(self.scene),next_id=1,objects={},frames={},schema=1)
  if self.state['scene']!=str(self.scene):raise ValueError('Output belongs to another scene')
  self.cal=self.calibration()
  if 'completed_frames' not in self.state:
   self.state['completed_frames']={}
   for i,ts in enumerate(self.frames):
    cams=[c for c in range(6) if (self.scene/'mav0'/f'cam{c}'/'data'/(ts+'.png')).exists()]
    if cams and all((self.out/'exports'/f'cam{c}'/(ts+'.png')).exists() for c in cams):self.state['completed_frames'][str(i)]=True
  self.save()
 def save(self):atomic(self.out/'project.json',json.dumps(self.state,ensure_ascii=False,indent=2).encode())
 def calibration(self):
  f=self.scene/'calib/cameras.py';tree=ast.parse(f.read_text());n=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='extrinsics' for t in n.targets));d={'np':np};exec(compile(ast.Module(body=[n],type_ignores=[]),'cal','exec'),d);cal={}
  for file in (self.scene/'calib/Intrinsic').glob('*.yaml'):
   for v in yaml.safe_load(file.read_text()).values():
    if not isinstance(v,dict) or 'intrinsics' not in v:continue
    m=re.search(r'cam(\d)',v.get('rostopic',''))
    if not m:continue
    c=int(m.group(1));fx,fy,cx,cy=v['intrinsics'];rw,rh=v['resolution'];K=np.array([[fx*W/rw,0,cx*W/rw],[0,fy*H/rh,cy*H/rh],[0,0,1.]])
    cal[c]=(K,np.array(v['distortion_coeffs']),d['extrinsics'][c]['R'],d['extrinsics'][c]['t'])
  return cal
 def image(self,f,c):
  f=int(f);c=int(c)
  if not 0<=f<len(self.frames) or c not in range(6):raise ValueError('invalid frame/camera')
  key=(f,c)
  if key not in self.images:
   p=self.scene/'mav0'/f'cam{c}'/'data'/(self.frames[f]+'.png');im=cv2.imread(str(p)) if p.exists() else None
   if im is None:raise ValueError('该时间戳缺少此相机图片')
   self.images[key]=cv2.resize(im,(W,H))
   if len(self.images)>18:self.images.pop(next(iter(self.images)))
  return self.images[key]
 def entries(self,f,c):return self.state['frames'].get(str(f),{}).get(str(c),{})
 def mask(self,e):return cv2.imread(str(self.out/e['mask']),0)
 def put(self,f,c,obj,mask,origin,reviewed=False,kind="mask"):
  if int(mask.sum())==0:return
  if origin!='manual' and str(obj) in self.state.get('dismissed',{}).get(f'{f}:{c}',[]):return
  self.state.setdefault('completed_frames',{}).pop(str(f),None)
  name=f'masks/{f:06d}/cam{c}/{obj}_{time.time_ns()}.png';atomic(self.out/name,png(mask));self.state['frames'].setdefault(str(f),{}).setdefault(str(c),{})[str(obj)]=dict(mask=name,origin=origin,reviewed=reviewed,kind=kind)
 def refresh_exports(self,frames):
  for f in frames:
   if any((self.out/'exports'/f'cam{c}'/(self.frames[f]+'.png')).exists() for c in range(6)):self.export(f)
 def invalidate(self,f,obj):
  affected=[]
  for fi,cams in self.state['frames'].items():
   if int(fi)<=f:continue
   for entries in cams.values():
    e=entries.get(str(obj))
    if e and e['origin']!='manual':entries.pop(str(obj));affected.append(int(fi))
  return affected
 def add(self,d):
  f=int(d['frame']);c=int(d['cam']);self.image(f,c);kind=d.get('kind','mask')
  if kind=='box':
   b=np.asarray(d['box'],dtype=float)
   if b.shape!=(4,) or not np.isfinite(b).all():raise ValueError('invalid box')
   x0,x1=sorted(np.clip(b[[0,2]],0,W-1).astype(int));y0,y1=sorted(np.clip(b[[1,3]],0,H-1).astype(int))
   if x1-x0<3 or y1-y0<3:raise ValueError('矩形框太小')
   m=np.zeros((H,W),np.uint8);m[y0:y1+1,x0:x1+1]=255
  else:m=decode(d['mask'])
  obj=str(d.get('object',''))
  if not m.any():raise ValueError('区域为空，未保存')
  if not obj:
   cls=int(d['class_id'])
   if cls not in CLASSES:raise ValueError('unknown class')
   obj=str(self.state['next_id']);self.state['next_id']+=1;self.state['objects'][obj]=dict(class_id=cls,start=f,stop=None)
  if obj not in self.state['objects']:raise ValueError('unknown object')
  o=self.state['objects'][obj]
  if o['stop'] is not None and f>=o['stop']:raise ValueError('此目标已停止追踪，请新建目标')
  affected=self.invalidate(f,obj);self.put(f,c,obj,m,'manual',True,kind);self.save();self.refresh_exports(set(affected+[f]));return obj
 def delete_box(self,f,c,obj):
  obj=str(obj)
  if obj not in self.entries(f,c):raise ValueError('目标框不存在')
  self.entries(f,c).pop(obj);blocked=self.state.setdefault('dismissed',{}).setdefault(f'{f}:{c}',[])
  if obj not in blocked:blocked.append(obj)
  affected=[f]
  for fi,cams in self.state['frames'].items():
   e=cams.get(str(c),{}).get(obj)
   if int(fi)>f and e and e['origin']!='manual':cams[str(c)].pop(obj);affected.append(int(fi))
  self.save();self.refresh_exports(affected)
 def reassign_box(self,f,c,obj,new_id,class_id):
  obj=str(obj);new_id=str(new_id);class_id=int(class_id);e=self.entries(f,c).get(obj)
  if not e or class_id not in CLASSES:raise ValueError('目标或类别无效')
  if new_id:
   o=self.state['objects'].get(new_id)
   if not o or (o['stop'] is not None and f>=o['stop']):raise ValueError('目标ID不存在或已停止')
   if new_id!=obj and new_id in self.entries(f,c):raise ValueError('此相机已有该ID，不覆盖现有目标')
   if new_id!=obj and o['class_id']!=class_id:raise ValueError('已有ID必须使用其原类别')
  mask=self.mask(e).copy();kind=e.get('kind','mask')
  if new_id==obj:
   self.state['objects'][obj]['class_id']=class_id
   self.put(f,c,obj,mask,'manual',True,kind);affected=self.invalidate(f,obj);self.save();self.refresh_exports(set([int(i) for i in self.state['frames']]+affected));return obj
  self.delete_box(f,c,obj)
  ys,xs=np.where(mask>0)
  return self.add(dict(frame=f,cam=c,object=new_id,class_id=class_id,mask=encode(mask),kind=kind,box=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]))
 def stop(self,f,obj):
  obj=str(obj)
  if obj not in self.state['objects']:raise ValueError('unknown object')
  self.state['objects'][obj]['stop']=int(f)
  affected=[]
  for fi,cams in self.state['frames'].items():
   if int(fi)>=f:
    affected.append(int(fi))
    for entries in cams.values():entries.pop(obj,None)
  self.save();self.refresh_exports(affected)
 def temporal(self,f):
  if f<=0:return []
  warnings=[]
  for c in range(6):
   old=self.entries(f-1,c)
   if not old:continue
   try:a=self.image(f-1,c);b=self.image(f,c)
   except ValueError:continue
   if float(self.frames[f])-float(self.frames[f-1])>1:warnings.append(f'cam{c}: 时间间隔超过1秒，暂停追踪');continue
   ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
   for obj,e in list(old.items()):
    o=self.state['objects'][obj]
    if (o['stop'] is not None and f>=o['stop']) or obj in self.entries(f,c):continue
    mask=self.mask(e)
    try:
     pts=cv2.goodFeaturesToTrack(ga,150,.01,7,mask=mask)
     if pts is None or len(pts)<6:raise ValueError('目标纹理不足')
     q,ok,_=cv2.calcOpticalFlowPyrLK(ga,gb,pts,None)
     if q is None:raise ValueError('光流未找到对应')
     back,ok2,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,None)
     if back is None:raise ValueError('反向光流失败')
     valid=(ok[:,0]>0)&(ok2[:,0]>0)&(np.linalg.norm(back[:,0]-pts[:,0],axis=1)<1.5)
     if valid.sum()<6:raise ValueError('一致的光流点不足')
     A,ins=cv2.estimateAffinePartial2D(pts[valid],q[valid],method=cv2.RANSAC,ransacReprojThreshold=3)
     if A is None or ins.sum()<6 or ins.mean()<.5:raise ValueError('运动拟合不可靠')
     if not .65<np.linalg.norm(A[:,0])<1.5:raise ValueError('运动尺度异常')
     moved=cv2.warpAffine(mask,A,(W,H),flags=cv2.INTER_NEAREST);ys,xs=np.where(moved>0)
     if len(xs)<30:raise ValueError('目标移出视野')
     box=[max(0,int(xs.min())-4),max(0,int(ys.min())-4),min(W-1,int(xs.max())+4),min(H-1,int(ys.max())+4)]
     if e.get('kind')=='box':
      rectangle=np.zeros((H,W),np.uint8);rectangle[int(ys.min()):int(ys.max())+1,int(xs.min()):int(xs.max())+1]=255;self.put(f,c,obj,rectangle,'temporal_box',False,'box');continue
     refined,score=self.seg.predict(b,box);union=np.count_nonzero((refined>0)|(moved>0));iou=np.count_nonzero((refined>0)&(moved>0))/max(union,1)
     if score<.6 or iou<.25:raise ValueError('SAM与运动预测不一致')
     self.put(f,c,obj,refined,'temporal_sam',False)
    except Exception as ex:
     if e.get('kind')=='box':warnings.append(f'cam{c} 目标{obj}: 框追踪暂停（{ex}），请重画');continue
     # Recovery from the latest source mask; never reuse a stopped ID or copy an old mask.
     try:
      ys,xs=np.where(mask>0)
      if len(xs)<30:raise ValueError('源区域为空或太小')
      dx=max(12,int((xs.max()-xs.min())*.2));dy=max(12,int((ys.max()-ys.min())*.2));box=[max(0,int(xs.min())-dx),max(0,int(ys.min())-dy),min(W-1,int(xs.max())+dx),min(H-1,int(ys.max())+dy)]
      recovered,score=self.seg.predict(b,box);area=np.count_nonzero(recovered);overlap=np.count_nonzero((recovered>0)&(mask>0))/max(1,np.count_nonzero((recovered>0)|(mask>0)))
      if score<.7 or overlap<.2 or not .3<area/len(xs)<3:raise ValueError('局部恢复置信度或区域一致性不足')
      self.put(f,c,obj,recovered,'temporal_sam_recovery',False);warnings.append(f'cam{c} 目标{obj}: 已局部重新定位，请检查')
     except Exception as recovery:warnings.append(f'cam{c} 目标{obj}: 追踪暂停（{ex}；{recovery}），在当前帧重画同一ID后NEXT继续')
  self.save();return warnings
 def cross(self,f,source,obj):
  e=self.entries(f,source).get(str(obj))
  if not e:raise ValueError('源相机没有此目标标注')
  a=self.image(f,source);mask=self.mask(e);sift=cv2.SIFT_create(nfeatures=6000);ka,da=sift.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),mask);log=[]
  if da is None or len(ka)<8:return ['跨相机特征不足，未建立关联']
  Ki,di,Ri,ti=self.cal[source]
  for c in range(6):
   if c==source or str(obj) in self.entries(f,c) or c not in self.cal:continue
   try:
    b=self.image(f,c);kb,db=sift.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
    if db is None:continue
    bf=cv2.BFMatcher();pairs=bf.knnMatch(da,db,k=2);rev={v[0].queryIdx:v[0].trainIdx for v in bf.knnMatch(db,da,k=2) if len(v)==2 and v[0].distance<.75*v[1].distance};good=[v[0] for v in pairs if len(v)==2 and v[0].distance<.75*v[1].distance and rev.get(v[0].trainIdx)==v[0].queryIdx]
    if len(good)<8:continue
    x=np.float32([ka[m.queryIdx].pt for m in good]);y=np.float32([kb[m.trainIdx].pt for m in good]);Kj,dj,Rj,tj=self.cal[c];R=Rj.T@Ri;t=Rj.T@(ti-tj);tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]]);E=tx@R
    xn=cv2.undistortPoints(x[:,None],Ki,di)[:,0];yn=cv2.undistortPoints(y[:,None],Kj,dj)[:,0];xh=np.column_stack([xn,np.ones(len(x))]);yh=np.column_stack([yn,np.ones(len(y))]);l=xh@E.T;l2=yh@E;num=np.abs(np.sum(yh*l,axis=1));err=np.maximum(num/np.maximum(np.linalg.norm(l[:,:2],axis=1),1e-9),num/np.maximum(np.linalg.norm(l2[:,:2],axis=1),1e-9))*max(Ki[0,0],Kj[0,0]);good=err<3
    if good.sum()<8:continue
    target=y[good];xmin,ymin=target.min(0);xmax,ymax=target.max(0)
    if (xmax-xmin)*(ymax-ymin)<100:continue
    box=[max(0,xmin-12),max(0,ymin-12),min(W-1,xmax+12),min(H-1,ymax+12)];
    if e.get('kind')=='box':
     m=np.zeros((H,W),np.uint8);m[int(box[1]):int(box[3])+1,int(box[0]):int(box[2])+1]=255;self.put(f,c,obj,m,'cross_camera_box',False,'box');log.append(f'目标{obj} → cam{c}: 框关联候选');continue
    m,score=self.seg.predict(b,box,points=target.tolist(),labels=[1]*len(target))
    inside=np.mean(m[np.clip(target[:,1].astype(int),0,H-1),np.clip(target[:,0].astype(int),0,W-1)]>0)
    if score<.6 or inside<.8:continue
    self.put(f,c,obj,m,'cross_camera_suggestion',False);log.append(f'目标{obj} → cam{c}: 自动关联候选（需审核）')
   except Exception as ex:log.append(f'cam{c}: '+str(ex))
  self.save();return log or ['无可靠跨相机对应；可选择同一目标ID在另一相机手绘，手工关联']
 def first_unfinished(self):
  done=self.state.get('completed_frames',{})
  return next((i for i in range(len(self.frames)) if str(i) not in done),len(self.frames)-1)
 def complete(self,f):
  for c in range(6):
   for e in self.entries(f,c).values():e['reviewed']=True
  self.state.setdefault('completed_frames',{})[str(f)]=True;self.save()
 def data(self,f):
  cameras=[]
  for c in range(6):
   try:self.image(f,c);exists=True
   except ValueError:exists=False
   entries=[]
   for obj,e in self.entries(f,c).items():
    o=self.state['objects'][obj]
    if o['stop'] is not None and f>=o['stop']:continue
    m=self.mask(e);ys,xs=np.where(m>0);bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None
    entries.append(dict(id=obj,**o,**e,box=bbox,mask_data=encode(m)))
   cameras.append(dict(cam=c,exists=exists,entries=entries))
  return dict(frame=f,timestamp=self.frames[f],count=len(self.frames),cameras=cameras,objects=self.state['objects'],classes=CLASSES)
 def export(self,f):
  result=[]
  for c in range(6):
   entries=self.entries(f,c)
   p=self.scene/'mav0'/f'cam{c}'/'data'/(self.frames[f]+'.png');im=cv2.imread(str(p)) if p.exists() else None
   if im is None:continue
   h,w=im.shape[:2];label=np.full((H,W),255,np.uint8);inst=np.zeros((H,W),np.uint16)
   # Background/stuff first, instances last; deterministic ID precedence.
   for obj,e in sorted(entries.items(),key=lambda z:(self.state['objects'][z[0]]['class_id'] in [2,3],int(z[0]))):
    if not e['reviewed'] or e.get('kind')=='box':continue
    m=self.mask(e)>0;label[m]=self.state['objects'][obj]['class_id'];inst[m]=int(obj)
   boxes=[]
   for obj,e in entries.items():
    if e.get('kind')!='box' or not e['reviewed']:continue
    ys,xs=np.where(self.mask(e)>0)
    if len(xs):boxes.append(dict(id=int(obj),class_id=self.state['objects'][obj]['class_id'],xyxy=[float(xs.min()*w/W),float(ys.min()*h/H),float((xs.max()+1)*w/W),float((ys.max()+1)*h/H)]))
   folder=self.out/'exports'/f'cam{c}';atomic(folder/(self.frames[f]+'_boxes.json'),json.dumps(dict(width=w,height=h,boxes=boxes)).encode());atomic(folder/(self.frames[f]+'.png'),png(cv2.resize(label,(w,h),interpolation=cv2.INTER_NEAREST)));atomic(folder/(self.frames[f]+'_instance.png'),png(cv2.resize(inst,(w,h),interpolation=cv2.INTER_NEAREST)));result.append(str(folder))
  atomic(self.out/'exports/classes.json',json.dumps(dict(classes=CLASSES,ignore=255,working_resolution=[W,H],note='nearest-neighbor upsample to original size; unreviewed and unlabelled remain ignore'),ensure_ascii=False,indent=2).encode());return result
