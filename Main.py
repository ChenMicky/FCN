from data_preprocess import write_to_TFrecord,Read_TFrecord
import tensorflow as tf 
import FCN as FCN
train_path = "A:\\test_img_TFRecords\\img"
pattern = '\*.png'
Label_path = "A:\\test_img_TFRecords\\lab"
shuffle_data = True
save_path = 'A:\\test_img_TFRecords//.data.tfrecords'


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size","1 ","batch_size")
tf.flags.DEFINE_string("log_dir","C:/Users/micky/Desktop/Python_pratical/TF_praatical/FCN/Log",'log_path')
tf.flags.DEFINE_string("image_dir","A:\\test_img_TFRecords\\img",'images_data')
tf.flags.DEFINE_string("label_dir","A:\\test_img_TFRecords\\lab","label_data")
tf.flags.DEFINE_string("TFrecord_dir","A:\\test_img_TFRecords//.data.tfrecords","TFrecord")
tf.flags.DEFINE_bool('regulization', "False", "regulization mode: True/ False")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "C:/Users/micky/Desktop/Python_pratical/TF_praatical/FCN/imagenet-vgg-verydeep-19.mat", "Path to vgg model mat")
tf.app.flags.DEFINE_integer('img_resize',"48","""image_resize""")
tf.app.flags.DEFINE_integer('image_h', "360", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "480", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('NUM_OF_CLASSESS', "75", """ total class number """)
tf.app.flags.DEFINE_integer("max_steps","1000","max_step")





def data_to_TFrecord(FLAGS):
  ## transfer data to TFRecords
  
  TFrecord = write_to_TFrecord(FLAGS)
  TFrecord.Image_type = 'png'
  images,labels = TFrecord.read_file()
  
  TFrecord.transfer_to_TFrecord(images,labels)
  print("tranfering data to TFrecord is successful ")

def training (image,label_batch,FLAGS,is_finetune = False):
  max_steps =FLAGS.max_steps

  startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-'))

  
  keep_prbability = tf.placeholder(tf.float32,name= "keep_prbabilty")

  global_step = tf.Variable(0, trainable=False)
  pred_annotation, logits =FCN.inference(image,keep_prbability,FLAGS)
  ####
  tf.summary.image("input_image", image, max_outputs=2)
  tf.summary.image("ground_truth", tf.cast(label_batch, tf.uint8), max_outputs=2)
  tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
  

  ####
  labels =tf.squeeze(label_batch, squeeze_dims=[3])
  labels =tf.cast(labels,tf.int32)
  loss  =tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=labels,
                                                                          name="entropy")))
  loss_summary = tf.summary.scalar("entropy",loss)

  train_op =FCN.train(loss,global_step,FLAGS)
  summary_op = tf.summary.merge_all()
  saver = tf.train.Saver()
  with tf.Session() as sess:
    if (is_finetune == True):
      ckpt= tf.train.get_checkpoint_state(FLAGS.logs_dir)
      if ckpt and ckpt.model_checkpoint_path :
        saver.restore (sess,ckpt.model_checkpoint_path )
      print("Model restored ..")
    else:
      init = tf.global_variables_initializer()
      sess.run(init)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_writer =tf.summary.FileWriter(FLAGS.log_dir+'/train',sess.graph)
    validation_writer  = tf.summary.FileWriter(FLAGS.logdir+'/val')

    for step in range(0, FLAGS.max_steps):
      _,loss_value = sess.run([train_op,loss],feed_dict={keep_prbability:0.85})
      if step % 10 ==0:
        print("step:%d,Train_loss:%g"%(step,loss_value))

    



    coord.request_stop()
    coord.join(threads)
    
def main(args):
  read_tfrecord =Read_TFrecord(FLAGS)
  ##next_element=read_tfrecord.Dataset_read()
  img_batch_data ,label_batch_data  =read_tfrecord.Dataset_read()
  training(img_batch_data,label_batch_data,FLAGS,is_finetune=False)

if __name__ == '__main__':
  tf.app.run()

  