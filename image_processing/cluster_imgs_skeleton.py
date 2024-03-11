import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import networkx as nx
import sys
sys.path.append('../search_algorithm')
from dfs import DFS

img_name = 'me_out'
img_dir = '../imgs/'
max_width = 11 # in pixels
min_stroke_length = 20 # in pixels
min_white_pixels = 50
save_paths = True

# Read image
image_path = Path(img_dir+img_name+'.png')
image = cv2.imread(str(image_path))
## convert image to gray
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## thresholding
_, image_thresh = cv2.threshold(image_gray, 15, 255, cv2.THRESH_BINARY)
# resize image
size_ratio = 1.5
print("Original image size: ", image_thresh.shape)
image_thresh = cv2.resize(image_thresh, (int(image_thresh.shape[1]*size_ratio), int(image_thresh.shape[0]*size_ratio)), interpolation = cv2.INTER_NEAREST)
print("Resized image size: ", image_thresh.shape)
# show image
plt.imshow(image_thresh, cmap='gray')
plt.show()

image_filled = deepcopy(image_thresh)
image_vis = deepcopy(image_thresh)*2/3

strokes_split = []
count=1
while True:
    ## invert image_thresh
    image_thresh_flip = cv2.bitwise_not(image_filled)
    ## skeletonize
    image_skeleton = cv2.ximgproc.thinning(image_thresh_flip)
    plt.imshow(image_vis+image_skeleton, cmap='gray')
    # plt.imshow(image_filled, cmap='gray')
    plt.show()

    ## find the distance closest black pixel in image_thresh using distance transform
    dist_transform = cv2.distanceTransform(image_thresh_flip, cv2.DIST_L2, 5)

    ## find white pixels in image_skeleton and loop through them
    white_pixels = np.where(image_skeleton == 255)
    white_pixels_removed = []
    image_viz_boarder = deepcopy(image_skeleton)
    
    if count>0:
        image_skeleton=np.zeros_like(image_skeleton)
    for i in range(len(white_pixels[0])):
        x = white_pixels[1][i]
        y = white_pixels[0][i]
        if dist_transform[y,x]>1:
            white_pixels_removed.append([y,x])
            if count>0:
                image_skeleton[y,x]=255
        image_viz_boarder[y,x]=255-dist_transform[y,x]

    white_pixels = np.where(image_skeleton == 255)
    
    # plt.imshow(image_vis+image_viz_boarder, cmap='gray')
    # plt.show()
    
    plt.imshow(image_vis+image_skeleton, cmap='gray')
    plt.show()

    edge_count = 0
    edges = []
    for i in range(len(white_pixels[0])):
        x = white_pixels[1][i]
        y = white_pixels[0][i]
        white_n = np.sum(image_skeleton[y-1:y+2, x-1:x+2] > 0)
        if white_n==2:
            edge_count+=1
            edges.append([x, y])
    print(f"Number of white pixels with 2 white neighbors: {edge_count}")
    print(f"Number of white pixels: {len(white_pixels[0])}")
    if len(white_pixels[0])<min_white_pixels:
        print("Number of white pixels is less than min_white_pixels")
        break

    ## find min max stroke width
    max_dist = max(dist_transform[white_pixels])
    min_dist = min(dist_transform[white_pixels])
    print(f"Max distance: {max_dist}, Min distance: {min_dist}")

    for i in range(len(white_pixels[0])):
        x = white_pixels[1][i]
        y = white_pixels[0][i]
        # if dist_transform[y][x]<=max_width:
        image_filled = cv2.circle(image_filled, (x, y), min(max(int(dist_transform[y][x]),1),max_width), 255, -1)
        image_vis = cv2.circle(image_vis, (x, y), min(max(int(dist_transform[y][x]),1),max_width), 120, -1)
    
    ## find strokes with deep first search, starting from the edge pixels
    plt.imshow(image_skeleton, cmap='gray')
    plt.show()
    
    dfs = DFS(image_skeleton, edges)
    strokes = dfs.search(from_edge=True)
    ## split strokes into segments, and find the width of each segment
    for m in range(len(strokes)):
        indices=[]
        widths=[]
        for i in range(len(strokes[m])-1):
            if np.linalg.norm(strokes[m][i]-strokes[m][i+1])>2:
                indices.append(i+1)
            # get the width of the point on the stroke
            widths.append(dist_transform[strokes[m][i][1],strokes[m][i][0]])
        widths.append(dist_transform[strokes[m][-1][1],strokes[m][-1][0]])
        
        #split path
        path_split=np.split(strokes[m],indices)
        widths_split=np.split(widths,indices)
        for i in range(len(path_split)):
            if len(path_split[i])<min_stroke_length:
                print("length too short")
                continue
            strokes_split.append(np.hstack((path_split[i],widths_split[i].reshape(-1,1))))
    
    
    # ### construct using networkx
    # directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
    # white_pixels=np.array(white_pixels).T
    # graph=nx.Graph()
    # draw_graph_pos={}
    # for i in range(len(white_pixels)):
    #     x = white_pixels[i][1]
    #     y = white_pixels[i][0]
    #     graph.add_node((x,y),width=dist_transform[y][x])
    #     draw_graph_pos[(x,y)]=(x,y)
    #     for direction in directions:
    #         x_ = x+direction[0]
    #         y_ = y+direction[1]
    #         if 0 <= x_ < image_skeleton.shape[1] and 0 <= y_ < image_skeleton.shape[0]:
    #             if image_skeleton[y_][x_] > 0:
    #                 graph.add_node((x_,y_),width=dist_transform[y_][x_])
    #                 graph.add_edge((x,y),(x_,y_),weight=np.linalg.norm([x-x_,y-y_]))
    # ## check isolation
    # graph.remove_nodes_from(list(nx.isolates(graph)))
    # print("Total nodes: ", graph.number_of_nodes(), "Total edges: ", graph.number_of_edges())
    # options = {
    # 'node_color': 'black',
    # 'node_size': 10,
    # }  
    # nx.draw(graph, draw_graph_pos, **options)
    # plt.show()
    
    # # find subgraphs
    # subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    # total_g=len(subgraphs)
    # all_paths=[]
    # count=1
    # for subg in subgraphs:    
    #     print("graph: ",count,"/",total_g," nodes: ",subg.number_of_nodes(), " edges: ", subg.number_of_edges())
    #     if subg.number_of_nodes()<min_stroke_length:
    #         print("length too short")
    #         continue
    #     draw_path = nx.approximation.traveling_salesman_problem(subg, cycle=False)
    #     all_paths.append(draw_path)
    #     count+=1
    # print("End tsp...")
    
    count+=1
strokes = strokes_split

if save_paths:
    ## save to strokes to file
    Path('../path/pixel_path/'+img_name+'/').mkdir(parents=True, exist_ok=True)
    for i in range(len(strokes)):
        np.savetxt('../path/pixel_path/'+img_name+'/'+str(i)+'.csv', strokes[i], delimiter=',')
    ## save resized image
    cv2.imwrite('../imgs/'+img_name+'_resized.png', image_thresh)

image_out = np.ones_like(image_thresh_flip)*255
for stroke in strokes:
    for n in stroke:
        image_out = cv2.circle(image_out, (int(n[0]), int(n[1])), round(n[2]), 0, -1)
        cv2.imshow("Image", image_out)
        cv2.waitKey(0)  