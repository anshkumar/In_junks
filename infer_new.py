#############################
# author: Vedanshu
#############################

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import cv2, queue, threading, time
import time
from IPython import embed

##############################
## Non-RT Graph conversion
##############################

detection_graph1 = tf.Graph()

with detection_graph1.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    tf_sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=detection_graph1)
    model = tf.saved_model.loader.load(tf_sess1, ["serve"], "/home/xavier/trt_l1")
    # model = tf.saved_model.loader.load(tf_sess1, ["serve"], "/home/xavier/mobilenet_l1/saved_model")

##############################
## RT Graph conversion
##############################

# output_names = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']
# trt_graph1 = trt.create_inference_graph(
#             input_graph_def=None,
#             outputs=output_names,
#             input_saved_model_dir='/home/xavier/mobilenet_l1/saved_model',
#             input_saved_model_tags=['serve'],
#             max_batch_size=400,
#             max_workspace_size_bytes=7000000000,
#             precision_mode='FP16') 

# detection_graph1 = tf.Graph()

# with detection_graph1.as_default():
#     tf.import_graph_def(trt_graph1, name='')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
#     tf_sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=detection_graph1)

input_names = ['image_tensor']
tf_input = tf_sess1.graph.get_tensor_by_name(input_names[0] + ':0')
tf_scores = tf_sess1.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess1.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess1.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess1.graph.get_tensor_by_name('num_detections:0')

#####################################
## Direct inference from saved_model
#####################################
# predict_fn = tf.contrib.predictor.from_saved_model("/home/xavier/mobilenet_l1/saved_model/")


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

#vcap = VideoCapture(0)
vcap = VideoCapture('http://10.42.0.211:8080/?action=stream')
'''

def gstreamer_pipeline (capture_width=2592, capture_height=1944, display_width=2592, display_height=1944, framerate=30, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

vcap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
# vcap = cv2.VideoCapture(0)
# vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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

while(True):
    # Capture frame-by-frame
    ret, frame = vcap.read()

    if frame is not None:
        # frame = _frame[200:1600 ,400:2000 ]
        # h, w = _frame.shape[:2]
        # data = np.load('calib.npz')
        # mtx = data['mtx']
        # dist = data['dist']
        # rvecs = data['rvecs']
        # tvecs = data['tvecs']
        # newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # frame = cv2.undistort(_frame, mtx, dist, None, newCameraMtx)
        
        cv2.circle(frame,( int(frame.shape[1]/2), int(frame.shape[0]/2) ), 5, (0,0,0), -1)
        cv2.circle(frame,( 1947, 624 ), 5, (0,0,0), -1)
        cv2.circle(frame,( 630, 667 ), 5, (0,0,0), -1)
        cv2.circle(frame,( 633, 1338 ), 5, (0,0,0), -1)
        cv2.circle(frame,( 1729, 1306 ), 5, (0,0,0), -1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print(img.shape)
        scores, boxes, classes, num_detections = tf_sess1.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={tf_input: img[None, ...]})

        # output_data = predict_fn({"inputs": img[None, ...]})
        # scores = output_data['detection_scores']
        # boxes = output_data['detection_boxes']
        # classes = output_data['detection_classes']
        # num_detections = output_data['num_detections']

        boxes = boxes[0]  # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = int(num_detections[0])

        # Boxes unit in pixels (image coordinates).
        boxes_pixels = []
        for i in range(num_detections):
            # scale box to image coordinates
            box = boxes[i] * np.array([frame.shape[0],
                               frame.shape[1], frame.shape[0], frame.shape[1]])
            box = np.round(box).astype(int)
            boxes_pixels.append(box)
        boxes_pixels = np.array(boxes_pixels)

        for i in range(num_detections):
             if scores[i] > 0.7:
               box = boxes_pixels[i]
               box = np.round(box).astype(int)

               if (box[3]-box[1])*(box[2]-box[0])*0.0079 < 44:
                 frame = cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
               else: 
                 frame = cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
               label = "{:.2f}:{:.2f}".format((box[3]-box[1])*0.35, (box[2]-box[0])*0.35)
               # label = "{}".format(i+1)
               # Draw label (class index and probability).
               draw_label(frame, (box[1], box[0]), label)
        
        cv2.putText(frame, "Total: "+str(num_detections), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.putText(frame, "Weight: "+str(154*num_detections)+" gms", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        label = "fps: {:.2f}: detections: {}".format(fps, num_detections)
        draw_label(frame, (10, 40), label)

        cv2.imshow(WINDOW_NAME,frame)
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

