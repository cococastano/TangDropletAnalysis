import cv2
import numpy as np
import data_utils
import extract_features


# just some example code

test_frame = data_utils.pull_frame_range(frame_range=[3],
                                         num_break=1, num_nobreak=1)
test_frame = test_frame[list(test_frame.keys())[0]][0]
width = test_frame.shape[0]
height = test_frame.shape[1]

# make a video
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#video = cv2.VideoWriter('all_frames_mark_up.avi', fourcc,10,(width*2,height*2))
#video = cv2.VideoWriter('all_frames_mark_up.avi',-1,20,(width*2,height*2))

n = 3  # number of vertices of polygon to interrogate
frame_range = [k for k in range(0,21)]
frames = data_utils.pull_frame_range(frame_range=frame_range,#frame_range,
                                     num_break=5, num_nobreak=5)
# do some plotting to verify what we've got so far
for frame_key in frames:
    for i, frame in enumerate(frames[frame_key]):      
        # add contours to show_frame
        show_frame = data_utils.show_my_countours(frame,contours=-1,
                                                  resize_frame=1,show=False)
        # add centroids to show_frame
        centroids, _ = extract_features.get_droplets_centroids(frame)
        for c in centroids:
            cX = centroids[c][0]
            cY = centroids[c][1]
            cv2.circle(show_frame, (cX,cY), 1, (0,0,255), 7)
            cv2.putText(show_frame, str(c), (cX + 4, cY - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # add polygon with n vertices to show_frame
        leading_centroids, _ = \
            extract_features.get_n_leading_droplets_centroids(frame,n)
        print('area: ', extract_features.polygon_area(leading_centroids), 
              '\t angle: ', 
              extract_features.leading_angle(leading_centroids)*180/np.pi,
              '\t frame key: ', frame_key)
        leading_centroids = [(coord) for coord in leading_centroids.values()]
        leading_centroids.append(leading_centroids[0])
        leading_centroids = np.int32(np.array(leading_centroids))        
        cv2.polylines(show_frame, [leading_centroids], True, (255,60,255))
        # add constriction location to show_frame
        constric_loc = data_utils.find_constriction(frame)
        y1 = int(frame.shape[0]/3)
        y2 = int(frame.shape[0]/3*2)
        cv2.line(show_frame, (constric_loc, y1), 
                 (constric_loc, y2), (0,150,255), 2)
        frame_str = frame_key.split('_')[0]
        frame_str = frame_key + ', frame ' + str(i)
        # add frame label to show_frame
        show_frame = cv2.putText(show_frame, frame_str, 
                                 (show_frame.shape[1]-250,
                                 show_frame.shape[0]-10),
                                 cv2.FONT_HERSHEY_COMPLEX, 
                                 0.5, (0, 0, 0), 2)
        # resize show_frame
        show_frame = data_utils.resize_my_frame(frame=show_frame,
                                                scale_factor=2)
        _,comb_cont_frame = extract_features.outer_perimeter(frame,
                                                             return_frame=True,
                                                             n=3)
        # show show_frame
        # video.write(show_frame)
        cv2.imshow('mark up', show_frame)
        cv2.imshow('combined contour', comb_cont_frame)
        cv2.waitKey(500)
	print('changes')
