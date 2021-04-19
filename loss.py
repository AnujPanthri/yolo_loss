from utils import intersection_over_union
import sys
import keras.backend as K


def custom_loss(y_true,y_pred):
###y_pred=[no. of images, 5 , 5 , 7] in which  7=>pc,x,y,w,h,c1,c2    5,5 are cells
  y_true_class=y_true[...,5:]
  y_pred_class=y_pred[...,5:]


  y_true_conf=y_true[...,0:1]
  y_pred_conf=y_pred[...,0:1]

  obj=y_true[...,0]==1            #true for cells which have objects
  noobj=y_true[...,0]==0          #true for cells which does not have objects

  #no obj loss
  noobj_loss=tf.reduce_mean(K.binary_crossentropy((y_true_conf[noobj]),(y_pred_conf[noobj]),from_logits=True))

  #obj loss
  box_pred=K.concatenate([K.sigmoid(y_pred_xy),(y_pred_wh)],axis=-1)
  ious=intersection_over_union(box_pred[obj],y_true[...,1:5][obj])
  obj_loss=tf.reduce_mean(K.binary_crossentropy((y_true_conf[obj]),(y_pred_conf[obj]),from_logits=True))

  #box cordinate loss
  box_loss=tf.reduce_mean(K.mean(K.square(K.concatenate([y_true[...,1:3],(y_true[...,3:5])])[obj]-K.concatenate([K.sigmoid(y_pred[...,1:3]),(y_pred[...,3:5])])[obj]), axis=-1))

  #class loss
  class_loss=tf.reduce_mean(K.categorical_crossentropy((y_true_class[obj]),(y_pred_class[obj]),from_logits=True))

  return ((0.8*noobj_loss)+(1.2*obj_loss)+(1*class_loss)+(5*box_loss)) 
