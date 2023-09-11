import numpy as np
import cv2
import os
import json
def get_bboxs(anot_i):
    bboxs = []
    for i in range(len(anot_i["cells"])):
        try:
            bboxs.append(anot_i["cells"][i]["bbox"])
        except:
            pass
    return bboxs

# anot_:
# {
#    'filename': str,
#    'split': str,
#    'imgid': int,
#    'html': {
#      'structure': {'tokens': [str]},
#      'cell': [
#        {
#          'tokens': [str],
#          'bbox': [x0, y0, x1, y1]  # only non-empty cells have this attribute
#        }
#      ]
#    }
# }
# N rows, M columns
def split_n_rotate(img_, col_separator_ranges, row_separator_ranges, N=300, M=200, angle=0): 
    # bboxs = get_bboxs(anot_)

    h, w, _ = img_.shape
    # w_c, w_r = w/M, h/N

    if angle == 0:
        ind_col, ind_row = 0, 0
        ind_cols, ind_rows = [0]*M, [0]*N

        c_, r_ = [], []
        for j in range(M):
            sep = int(j*w/M)
            separation = 0
            for i in range(ind_col, len(col_separator_ranges)):
                sep_range = col_separator_ranges[i]
                if sep>=sep_range[0] and sep<=sep_range[1]:
                    separation = 1
                    ind_cols[j] = i
                    ind_col = i
                    break
            c_.append(separation)

        for j in range(N):
            sep = int(j*h/N)
            separation = 0
            for i in range(ind_row, len(row_separator_ranges)):
                sep_range = row_separator_ranges[i]
                if sep>=sep_range[0] and sep<=sep_range[1]:
                    separation = 1
                    ind_rows[j] = i
                    ind_row = i
                    break   
            r_.append(separation)
        r_anot = np.array([r_, [0]*N, [0]*N, ind_rows]).T
        c_anot = np.array([c_, [0]*M, [0]*M, ind_cols]).T
        return img_, {"row":r_anot, "col":c_anot}


    #rotate
    height, width = img_.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(img_, rotation_mat, (bound_w, bound_h), borderValue=(255,255,255))

    # labeling
    # b = np.sqrt(np.power(width/2,2) + np.power(height/2,2))
    # c = np.sqrt(np.power(bound_w/2,2) + np.power(bound_h/2,2))
    # resid = np.sqrt(b**2 + c**2 - 2*b*c*np.cos(angle*np.pi/180))
    n,m = len(row_separator_ranges), len(col_separator_ranges)
    theta = np.abs(angle*np.pi/180)
    x = np.sin(theta)*height
    y = np.sin(theta)*width
    l_pad = int(np.tan(theta)*y)
    b_pad = int(np.tan(theta)*(x + l_pad))
    # c_shift = int(x/w_c)
    y = y/np.cos(theta)/np.cos(theta)
    # r_shift = int(y/w_r)

    if (angle > 0):
        # c_ = c_ + [0]*c_shift
        # r_ = [0]*r_shift + r_
        rotated_mat= cv2.copyMakeBorder(rotated_mat.copy(),0,b_pad,l_pad,0,cv2.BORDER_CONSTANT,value=[255,255,255])
    else:
        # c_ = [0]*c_shift + c_
        # r_ = r_ + [0]*r_shift
        rotated_mat= cv2.copyMakeBorder(rotated_mat.copy(),b_pad,0,0,l_pad,cv2.BORDER_CONSTANT,value=[255,255,255])

    ind_col, ind_row = 0, 0
    
    new_c, new_r = [0]*M, [0]*N
    ind_cols, ind_rows = [-1]*M, [-1]*N
    r_offsets, r_thetas = [-1]*N ,[90]*N
    c_offsets, c_thetas = [-1]*M ,[90]*M
    pad_x, pad_y = height*np.tan(theta), width*np.tan(theta) 
    for j in range(M):
        # s_point = int(j*bound_w/M)
        # e_point = int((j+1)*bound_w/M)
        # separation = 1
        # pos, neg = 0, 0
        # for ol in range(int(s_point/w_c), int(e_point/w_c)+1):
        #     try:
        #         if c_[ol] == 1: pos += 1
        #         else:   neg += 1
        #     except:
        #         neg += 1
        # if neg/(neg+pos) > 0.3: separation=0
        # new_c.append(separation)
        if ind_col>=m:
            break
        query = int(j*bound_w/M)
        sep = (query/np.cos(theta)-pad_x) if angle<0 else w-((bound_w-query)/np.cos(theta)-pad_x)
        separation = 0
        for i in range(ind_col, len(col_separator_ranges)):
            sep_range = col_separator_ranges[i]
            if sep>=sep_range[0] and sep<=sep_range[1]:
                separation = 1
                ind_cols[j] = i
                ind_col = i
                c_offsets[j] = (sep+pad_x)*np.sin(theta) if angle<0 else (w-sep+pad_x)/np.sin(theta)
                c_thetas[j] = angle
                break
        new_c[j] = separation
                

    for j in range(N):
        if ind_row>=n:
            break
        query = int(j*bound_h/N)
        sep = (query/np.cos(theta)-pad_y) if angle<0 else h-((bound_h-query)/np.cos(theta)-pad_y)
        separation = 0
        for i in range(ind_row, len(row_separator_ranges)):
            sep_range = row_separator_ranges[i]
            if sep>=sep_range[0] and sep<=sep_range[1]:
                separation = 1
                ind_rows[j] = i
                ind_row = i
                r_offsets[j] = (sep+pad_y)*np.sin(theta) if angle<0 else (w-sep+pad_y)/np.sin(theta)
                r_thetas[j] = angle
                break
        new_r[j] = separation       
        # e_point = int((j+1)*bound_h/N)
        # separation = 1
        # pos, neg = 0, 0
        # for ol in range(int(s_point/w_r), int(e_point/w_r)+1):
        #     try:
        #         if r_[ol] == 1: pos += 1
        #         else:   neg += 1
        #     except:
        #         neg += 1
        # if neg/(neg+pos) > 0.2: separation=0
        # new_r.append(separation)
    
    
    # for j in range(N):
    #     if new_r[j] == 1:
    #         r_thetas[j] = angle
    #         if angle > 0:
    #             r_offsets[j] = j*bound_h/N*np.tan(theta)
    #         else:
    #             r_offsets[j] = (bound_h - y - j*bound_h/N)*np.tan(theta)

    
    # for j in range(M):
    #     if new_c[j] == 1:
    #         c_thetas[j] = angle
    #         if angle < 0:
    #             c_offsets[j] = j*bound_w/M*np.tan(theta)
    #         else:
    #             c_offsets[j] = (bound_w - x - j*bound_w/M)*np.tan(theta)     
            



    return rotated_mat, {"row": np.array([new_r, r_offsets, r_thetas, ind_rows]).T, "col": np.array([new_c, c_offsets, c_thetas, ind_cols]).T}

path = "\\sample1.json"
cwd = os.getcwd()
anno_list = []
N, M, angle = 100, 100, 10
with open(cwd+path) as f:
    anno_list = json.load(f)
    # for object in f:
    #     anno = json.loads(object)
    #     anno_list.append(anno)
for img_anno in anno_list:
    img = cv2.imread('C:\\Users\\Admin\\Desktop\\table detection\\TRUST\\PubTabNet\\examples\\'+img_anno['filename'])
    col_separator_ranges = img_anno['col_separators']
    row_separator_ranges = img_anno['row_separators']
    rotated_img, trust_anno = split_n_rotate(img, col_separator_ranges, row_separator_ranges, N, M, angle)
    window_name = 'image'

    # displaying the groundtruth annotation
    print(trust_anno)
    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, rotated_img)
    
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    
    # closing all open windows
    cv2.destroyAllWindows()
    break