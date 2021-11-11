import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
import cv2
import matplotlib.pyplot as mpplt

def load_image(filename, DATA_ROOT):
  image_data = {}
  source = cv2.imread(DATA_ROOT+"source_"+filename) # source
  mask = cv2.imread(DATA_ROOT+"mask_"+filename) # mask
  target = cv2.imread(DATA_ROOT+"target_"+filename) # target
  
  
  image_data['source'] = cv2.normalize(source.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
  image_data['mask'] = cv2.normalize(mask.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
  image_data['target'] = cv2.normalize(target.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
  image_data['dims'] = [1, 1]
  
  return image_data

def display_image(image_data):
  mpplt.figure(figsize=(16,16))
  for i in range(3):
    if(i == 0):
      img_string = 'source'
    elif(i == 1):
      img_string = 'mask'
    else:
      img_string = 'target'
    img = image_data[img_string]
    mpplt.subplot(1,3,i+1)
    mpplt.imshow(img[:,:,[2,1,0]])
    
    
def preprocess(image_data):
  
  source = image_data['source']
  mask = image_data['mask']
  target = image_data['target']
  
  
  Hs,Ws,_ = source.shape
  Ht,Wt,_ = target.shape
  Ho, Wo = image_data['dims']
  
  
  if(Ho < 0):
    mask = np.roll(mask, Ho, axis=0)
    source = np.roll(source, Ho, axis=0)
    mask[Hs+Ho:,:,:] = 0 
    source[Hs+Ho:,:,:] = 0
    Ho = 0
  if(Wo < 0):
    mask = np.roll(mask, Wo, axis=1)
    source = np.roll(source, Wo, axis=1)
    mask[:,Ws+Wo:,:] = 0
    source[:,Ws+Wo:,:] = 0
    Wo = 0
  
  # mask region on target
  H_min = Ho
  H_max = min(Ho + Hs, Ht)
  W_min = Wo
  W_max = min(Wo + Ws, Wt)
  
  # crop source and mask if outside of target bounds
  source = source[0:min(Hs, Ht-Ho),0:min(Ws, Wt-Wo),:]
  mask = mask[0:min(Hs, Ht-Ho),0:min(Ws, Wt-Wo),:]
  
  return {'source':source, 'mask': mask, 'target': target, 'dims':[H_min,H_max,W_min,W_max]}


def naive_copy(image_data):
  # extract image data
  source = image_data['source']
  mask = image_data['mask']
  target = image_data['target']
  dims = image_data['dims']
  
  target[dims[0]:dims[1],dims[2]:dims[3],:] = target[dims[0]:dims[1],dims[2]:dims[3],:] * (1 - mask) + source * mask
  
  return target


def get_subimg(image, dims):
   return image[dims[0]:dims[1], dims[2]:dims[3]]

def poisson_blending(image, GRAD_MIX):
 
  def _compare(val1, val2):
    if(abs(val1) > abs(val2)):
      return val1
    else:
      return val2
  
 
  mask = image['mask']
  Hs,Ws = mask.shape
  num_pxls = Hs * Ws
  
  # source and target image
  source = image['source'].flatten(order='C')
  target_subimg = get_subimg(image['target'], image['dims']).flatten(order='C')

  
  mask = mask.flatten(order='C')
  guidance_field = np.empty_like(mask)
  laplacian = sps.lil_matrix((num_pxls, num_pxls), dtype='float64')

  for i in range(num_pxls):
    
    if(mask[i] > 0.2):
        
        if(mask[i] > 0.99):
            laplacian[i,i]=1
            Np_up_t = 0
            Np_left_t =0
            Np_down_t = 0
            Np_right_t = 0
            guidance_field[i] = source[i]
        else:
            laplacian[i, i] = 4
            if(i - Ws > 0):
                laplacian[i, i-Ws] = -1
                Np_up_s = source[i] - source[i-Ws]
                Np_up_t = target_subimg[i] - target_subimg[i-Ws]
            else:
                Np_up_s = source[i]
                Np_up_t = target_subimg[i]
            
            if(i % Ws != 0):
                laplacian[i, i-1] = -1
                Np_left_s = source[i] - source[i-1]
                Np_left_t = target_subimg[i] - target_subimg[i-1]
            else:
                Np_left_s = source[i]
                Np_left_t = target_subimg[i]
            
            if(i + Ws < num_pxls):
                laplacian[i, i+Ws] = -1
                Np_down_s = source[i] - source[i+Ws]
                Np_down_t = target_subimg[i] - target_subimg[i+Ws]
            else:
                Np_down_s = source[i]
                Np_down_t = target_subimg[i]
                
            if(i % Ws != Ws-1):
                laplacian[i, i+1] = -1
                Np_right_s = source[i] - source[i+1]
                Np_right_t = target_subimg[i] - target_subimg[i+1]
            else:
                Np_right_s = source[i]
                Np_right_t = target_subimg[i]
                
            
       
        
            guidance_field[i] = (_compare(Np_up_s, Np_up_t) + _compare(Np_left_s, Np_left_t) + 
                           _compare(Np_down_s, Np_down_t) + _compare(Np_right_s, Np_right_t))
           
    else : 
     
      laplacian[i, i] = 1
      guidance_field[i] = target_subimg[i]
      
  return [laplacian, guidance_field]


def linlsq_solver(A, b, dims):
  x = linalg.spsolve(A.tocsc(),b)
  return np.reshape(x,(dims[0],dims[1]))

# copie poisson solution -> target
def stitch_images(source, target, dims):
  target[dims[0]:dims[1], dims[2]:dims[3],:] = source
  return target

# poisson blending
def blend_image(data, GRAD_MIX):
  
    equation_param = []
    ch_data = {}
    
    # poisson equation 
    for ch in range(3):
      ch_data['source'] = data['source'][:,:,ch]
      ch_data['mask'] = data['mask'][:,:,ch]
      ch_data['target'] = data['target'][:,:,ch]
      ch_data['dims'] = data['dims']
      equation_param.append(poisson_blending(ch_data, GRAD_MIX))
    
    # solve equation
    image_solution = np.empty_like(data['source'])
    for i in range(3):
      image_solution[:,:,i] = linlsq_solver(equation_param[i][0],equation_param[i][1],data['source'].shape)
      
    image_solution = stitch_images(image_solution,image['target'],ch_data['dims'])
      
    return image_solution


DATA_ROOT = "Path//to//file"
GRAD_MIX = True
IMAGE_NAME = "test1.jpg"

image = load_image(IMAGE_NAME, DATA_ROOT)
display_image(image) # plot data
data = preprocess(image)
display_image(data)
final_image = blend_image(data,GRAD_MIX) # blend the image

# plot results
final_image = np.clip(final_image,0.0,1.0)
mpplt.subplot(1,3,3)
mpplt.imshow(final_image[:,:,[2,1,0]])

# save image
save_img = final_image * 255
save_img = save_img.astype(np.uint8)
cv2.imwrite(DATA_ROOT+'result_'+IMAGE_NAME, save_img, [cv2.IMWRITE_JPEG_QUALITY,90])

