#############################
# author: Vedanshu
#############################

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import cv2, queue, threading, time
import time
from IPython import embed
import servo_lib
import multiprocessing


def gstreamer_pipeline (capture_width=2592, capture_height=1944, display_width=2592, display_height=1944, framerate=30, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

detection_graph1 = tf.Graph()

with detection_graph1.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    tf_sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=detection_graph1)
    model1 = tf.saved_model.loader.load(tf_sess1, ["serve"], "/home/xavier/trt_l1")

# _graph2 = get_frozen_graph("/home/xavier/out/l2/frozen_inference_graph.pb")

detection_graph2 = tf.Graph()

with detection_graph2.as_default():
    # tf.import_graph_def(_graph2, name='')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
    tf_sess2 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=detection_graph2)
    model2 = tf.saved_model.loader.load(tf_sess2, ["serve"], "/home/xavier/trt_l2")

input_names = ['image_tensor']
tf_input1 = tf_sess1.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores1 = tf_sess1.graph.get_tensor_by_name('detection_scores:0')
tf_boxes1 = tf_sess1.graph.get_tensor_by_name('detection_boxes:0')
tf_classes1 = tf_sess1.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections1 = tf_sess1.graph.get_tensor_by_name('num_detections:0')

tf_input2 = tf_sess2.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores2 = tf_sess2.graph.get_tensor_by_name('detection_scores:0')
tf_boxes2 = tf_sess2.graph.get_tensor_by_name('detection_boxes:0')
tf_classes2 = tf_sess2.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections2 = tf_sess2.graph.get_tensor_by_name('num_detections:0')

# output_names = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']
# predict_fn = tf.contrib.predictor.from_saved_model("/home/xavier/ckpt/saved_model/")
#pb_fname2 = "/home/xavier/out/l2/frozen_inference_graph.pb"

class_lst = ['tomato', 'tomato spots holes', 'tomato tip', 'tomato cuts cracks', 'tomato shrivelled', 'tomato glare', 'tomato back tip', 'tomato stalk', 'tomato green area', 'tomato non black spots', "tomato rotten", "tomato water layer", "tomato water mark", "tomato deformed double", "tomato unripe", "tomato normal", 
    'onion', 'onion smut effected', 'onion without skin', 'onion sprouting', 'onion rotten', 'onion half cut', 'onion tip', 'onion neck', 'onion sun burn', "onion glare", "onion double onion", "onion open neck", "onion blurred", 
    'potato', 'potato big crack and cuts', 'potato decayed', 'potato sprouted', 'potato major hole', 'potato shriveled', 'potato greening', 'potato small crack and cuts', "potato badly deformed", "potato eyes", "potato confusing", "potato dried sprout"]

defect_id = [1]

'''
# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except Queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
  
  def release(self):
    self.cap.release()
'''
# vcap = VideoCapture('http://10.42.0.211:8080/?action=stream')
vcap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
# vcap = cv2.VideoCapture(0)
# vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
# vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)

def getSquareImage( img, target_width = 500 ):
    width = img.shape[1]
    height = img.shape[0]

    square = np.zeros( (target_width, target_width, 3), dtype=np.uint8 )

    max_dim = width if width >= height else height
    scale = target_width / max_dim
    
    if ( width >= height ):
        width = target_width
        x = 0
        height = int(height * scale)
        y = int(( target_width - height ) / 2)
    else:
        y = 0
        height = target_width
        width = int(width * scale)
        x = int(( target_width - width ) / 2)
    # embed()
    square[y:y+height, x:x+width] = cv2.resize( img , (width, height) )

    return square, x, y, scale

fps = 0.0
full_scrn = False
WINDOW_NAME = "IntelloLabs"
tic = time.time()
    
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

servo_control = servo_lib.servo()

while(True):
    # Capture frame-by-frame
    ret, frame = vcap.read()
    #print cap.isOpened(), ret
    if frame is not None:
        _img = frame[200:1000 ,200:1400 ]
        img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        scores, boxes, classes, num_detections = tf_sess1.run([tf_scores1, tf_boxes1, tf_classes1, tf_num_detections1], feed_dict={tf_input1: img[None, ...]})
        
        # output_data = predict_fn({"inputs": img[None, ...]})

        # scores = output_data['detection_scores']
        # boxes = output_data['detection_boxes']
        # classes = output_data['detection_classes']
        # num_detections = output_data['num_detections']

        boxes = boxes[0]  # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = int(num_detections[0])

        clone = frame.copy()

        for i in range(num_detections):
            if scores[i] > 0.5:
                # box = boxes_pixels[i]
                # box = np.round(box).astype(int)
                box = np.round(boxes[i] * np.array([_img.shape[0],_img.shape[1], _img.shape[0], _img.shape[1]])).astype(int)
                (startY, startX, endY , endX) = box.astype("int")
                startX = startX + 200
                startY = startY + 200
                endX = endX + 200
                endY = endY + 200
                boxW = endX - startX
                boxH = endY - startY
                roi = clone[startY:endY, startX:endX]
                roi_img, _x, _y, _scale = getSquareImage(roi, 300)
                # _img_batch.append(_img)

                # visMask = (mask * 255).astype("uint8")
                # print("visMask shape: ", visMask.shape)
                # print("roi shape: ", roi.shape)
                #embed()
                # instance = cv2.bitwise_and(roi, roi, mask=visMask)
                defective = False
                
                _scores, _boxes, _classes, _num_detections = tf_sess2.run([tf_scores2, tf_boxes2, tf_classes2, tf_num_detections2], feed_dict={tf_input2: roi_img[None, ...]})
                _boxes = _boxes[0]  # index by 0 to remove batch dimension
                _scores = _scores[0]
                _classes = _classes[0]
                _num_detections = int(_num_detections[0])
                
                for i in range(_num_detections):
                    if _scores[i] > 0.1:
                        if _classes[i] in defect_id:
                            defective = True
                        # _box = np.round(_boxes[i] * np.array([roi.shape[0],roi.shape[1], roi.shape[0], roi.shape[1]])).astype(int)
                        # _img = cv2.rectangle(_img, (_box[1] + startX - 200, _box[0] + startY - 200), (_box[3] + startX - 200, _box[2] + startY - 200), (255, 0, 0), 2)
                        # _label = "{}:{}".format(_classes[i], roi.shape)
                        # draw_label(_img, (_box[1] + startX - 200, _box[0] + startY - 200), _label)
                
                # Correcting bounding box
                box[1] = box[1] #+ 200
                box[0] = box[0] #+ 200
                box[3] = box[3] #+ 200
                box[2] = box[2] #+ 200
                # Draw bounding box.
                if defective == True:
                    _img = cv2.rectangle(_img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
                else:
                    _img = cv2.rectangle(_img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

                cv2.putText(_img, str(box[1]), (box[3], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
                if box[1] > 40 and box[3] < _img.shape[1] - 40:
                   if ((box[1] >=290) and (box[1]<=460)):
                        print('row 1')
                        if defective == True:
                            px2in = 22.7/1944  #px2in = 24.7/1944
                            _time = ((box[0])*px2in + 32)/8 # _time = round((box[1]*px2in + 32)/18)
                            another_process = multiprocessing.Process(target=servo_control.reject_tomatoe, args=(0,_time-0.5,))
                            another_process.start()
                            print(_time)
                            cv2.putText(_img, str(_time), (box[3], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

                   elif ((box[1] >=500) and (box[1]<=680)):
                        print('row 2')
                        if defective == True:
                            px2in = 22.7/1944  #px2in = 24.7/1944
                            _time = ((box[0])*px2in + 32)/8 #_time = round((box[1]*px2in + 32)/18)
                            another_process = multiprocessing.Process(target=servo_control.reject_tomatoe, args=(1,_time-0.5,))
                            another_process.start()
                            print(_time)
                            cv2.putText(_img, str(_time), (box[3], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

                   elif ((box[1] >=720) and (box[1]<=920)):
                        print('row 3')
                        if defective == True:
                            px2in = 22.7/1944  #px2in = 24.7/1944
                            _time = ((box[0])*px2in + 32)/8 #_time = round((box[1]*px2in + 32)/18)
                            another_process = multiprocessing.Process(target=servo_control.reject_tomatoe, args=(2,_time-0.5,))
                            another_process.start()
                            print(_time)
                            cv2.putText(_img, str(_time), (box[3], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

                   elif ((box[1] >=940) and (box[1]<=1136)):
                        print('row 4')
                        if defective == True:
                            px2in = 22.7/1944  #px2in = 24.7/1944
                            _time = ((box[0])*px2in + 32)/8 #_time = round((box[1]*px2in + 32)/18)
                            another_process = multiprocessing.Process(target=servo_control.reject_tomatoe, args=(3,_time-0.5,))
                            another_process.start()
                            print(_time)
                            cv2.putText(_img, str(_time), (box[3], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

                #embed()
                #label = "{}:{:.2f}".format(int(classes[i]), scores[i])
                # Draw label (class index and probability).
                #draw_label(frame, (box[1], box[0]), label)
        
        #Rotating image
        # row,col,_ = frame.shape
        # center=tuple(np.array([row,col])/2)
        # rot_mat = cv2.getRotationMatrix2D(center,180,1.0)
        # frame = cv2.warpAffine(frame, rot_mat, (row,col))

        label = "fps: {:.2f}".format(fps)
        draw_label(_img, (10, 40), label)

        cv2.imshow(WINDOW_NAME,_img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.9 + curr_fps*0.1)
        tic = toc
        print("fps = ", fps)
        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        print("Frame is None")
        break

# When everything done, release the capture
vcap.release()
cv2.destroyAllWindows()
print("Video stop")

