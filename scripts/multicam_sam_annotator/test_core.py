import tempfile,unittest
from pathlib import Path
import numpy as np,cv2
from core import Project,encode
SCENE='/home/ivan/project/IJRR/scene_0'
class Tests(unittest.TestCase):
 def test_lifecycle(self):
  with tempfile.TemporaryDirectory() as d:
   p=Project(SCENE,d,None);m=np.zeros((768,1024),np.uint8);m[200:260,300:360]=255
   obj=p.add(dict(frame=0,cam=0,mask=encode(m),class_id=2));p.put(1,0,obj,m,'temporal_sam');p.put(1,3,obj,m,'cross_camera_suggestion');p.save();p.export(1)
   label=cv2.imread(str(Path(d)/'exports/cam0'/f'{p.frames[1]}.png'),0);self.assertTrue(np.all(label==255))
   p.add(dict(frame=0,cam=0,mask=encode(m),object=obj));self.assertNotIn(obj,p.entries(1,0));self.assertNotIn(obj,p.entries(1,3))
   p.put(1,0,obj,m,'temporal_sam');p.stop(1,obj);self.assertIn(obj,p.entries(0,0));self.assertNotIn(obj,p.entries(1,0));self.assertEqual(p.state['objects'][obj]['stop'],1)
   with self.assertRaises(ValueError):p.add(dict(frame=1,cam=0,mask=encode(m),object=obj))
   p.export(0);label=cv2.imread(str(Path(d)/'exports/cam0'/f'{p.frames[0]}.png'),0);self.assertEqual(label[450,650],2);self.assertEqual(label[0,0],255)
   q=Project(SCENE,d,None);self.assertEqual(q.state['objects'][obj]['stop'],1)
if __name__=='__main__':unittest.main()
