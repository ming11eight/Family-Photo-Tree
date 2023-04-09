import os
import os.path
from collections import Counter
from sklearn import neighbors
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from detection import mtcnn_detect


def createFolder(directory:str):
    '''
        directory가 존재하면 생성하지 않고, 없다면 생성
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        

def save_crop_img(df, image, name):
    '''
        crop한 이미지 저장
    '''
    cnt = Counter(df['class'])[name]-1 # .jpg 파일 이름에 추가되는 변수
    # actor인 경우
    actor_path = f'./actor/{name}'
    createFolder(actor_path) # 폴더가 없다면 생성, 있다면 패스
    image.save(actor_path+f"/{name}_{cnt}.jpg") # actor폴더 하단에 저장


def changeFileName(src, pre_name, new_name):
    '''
        파일 이름 변경
    '''
    i=0
    for j in os.listdir(src+pre_name):
        os.rename(src+pre_name+'/'+j,src+pre_name+'/'+new_name+f'_{i}.jpg')


def train(model,device, mtcnn, train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    '''
        knn모델 학습을 위한 함수
        {model} - FaceNet Model 사용
        {mtcnn} - facenet_pytorch의 MTCNN 모듈 사용
        {train_dir} - knn모델 업데이트시, 학습할 데이터(crop되어 저장된 이미지)

    '''
    X = []
    y = []

    # Loop through each person in the training set
    train_dir = Path(train_dir)
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(train_dir/class_dir):
            continue

        # Loop through each training image for the current person
        for img_path in os.listdir(train_dir/class_dir):
            full_file_path = str(train_dir/class_dir/img_path)
            orig_image = Image.open(full_file_path)
            image = np.array(orig_image)
            face_bounding_boxes, boxes = mtcnn_detect(image,mtcnn,orig_box=True)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add fad(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                faces = mtcnn.extract(image, boxes, save_path=None)
                x_aligned=[]
                x_aligned.append(faces)
                aligned = torch.stack(x_aligned).to(device)
                embeddings = model(aligned).detach().cpu()
                
                encoding_new = []
                for e in embeddings:
                    encoding_new.append(np.array(e))
                encoding_new = np.array(encoding_new).flatten()
                X.append(encoding_new)
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, model, device, mtcnn, knn_clf=None, model_path=None, distance_threshold=0.6):
    '''
        knn 모델 predict를 위한 함수
    '''
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations   
    orig_image = Image.open(X_img_path)
    image = np.array(orig_image)
    X_face_locations, boxes = mtcnn_detect(image,mtcnn,orig_box=True)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    encoding_new = []
    are_matches = []
    # print(boxes)
    for box in boxes:
        box = np.array([box])
        # print(box)
        faces = mtcnn.extract(image, box, save_path=None)
    
        x_aligned=[]
        x_aligned.append(faces)
        aligned = torch.stack(x_aligned).to(device)
        embeddings = model(aligned).detach().cpu()
    
        for e in embeddings:
            encoding_new.append(np.array(e))
    # print("shape: ", np.array(encoding_new).shape)
    # print(encoding_new)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(encoding_new, n_neighbors=1)
    # print(closest_distances)

    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(encoding_new), X_face_locations, are_matches)]


def creat_df(save_file_name:str):
    ini_df = pd.DataFrame({'photo_ID':[], 'x_left':[], 'y_left':[],'x_right':[],'y_right':[],'class':[] })
    ini_df.to_csv(f"{save_file_name}.csv", index =False)
    df = pd.read_csv(f'./{save_file_name}.csv')
    return df


def check_to_value(img_cropped, name_dic:dict, df):
    '''
        name_dic에 value가 10인 key가 있는지 체크
    '''
    del_list=[]
    for k, v in name_dic.items():
        # 라벨 요청
        if v == 10:
            plt.imshow(img_cropped)
            plt.title(k)
            plt.show()
            print(f"There are 3 pictures of {k}")
            answer = input(f"Do you want to specify a label for {k}?[y/n] : ")
            if answer == 'y':
                name = input(f"Who is {k} : ")
                changeFileName('./actor/',k,name) # 파일 이름 변경 (unknown_?_? -> name_?)
                # if os.path.exists('./actor/'+name): # 기존에 존재하는 이름이라면
                #     for i in os.listdir('./actor/'+k): # unknwon_?/i
                #         unknown_name=i
                #         for exist_name in os.listdir('./actor/'+name):
                #             if i==exist_name:
                #                 x=str(exist_name).split('_')
                #                 n = x[-1].split('.')[0]
                #                 i = f'{x[0]}_{int(n)+1}.jpg'
                #             new_name = i
                #         os.rename('./actor/'+k+'/'+i,'./actor/'+name+'/'+new_name)
                #         shutil.move('./actor/'+k+'/'+new_name,'./actor/'+name)
                # else:
                os.rename('./actor/'+k,'./actor/'+name) # 폴더 이름 변경 (unknown_? -> name)
                shutil.rmtree('./extra/',k) # extra에 있는 폴더 삭제
                df = df.replace(k,name)  # df 업데이트
                del_list.append(k)
    for n in del_list:
        del(name_dic[n])  # name_dic 업데이트


def inference(model, device, mtcnn, input_dir:str, pic_name, orig_image, name_dic:dict, df):
    print("Looking for faces in {}".format(pic_name))
    predictions = predict(input_dir + pic_name, model, device, mtcnn, model_path="trained_knn_model.clf")

    # Print results on the console
    for name, (x_left, y_left, x_right, y_right) in predictions:
        print("- Found {}".format(name))
        img_cropped = orig_image.crop((x_left, y_left, x_right, y_right))
        
        # extra가 있는 경우
        if 'unknown' in name:
            # 모르는 얼굴('unknown')이 있는 경우
            if name == 'unknown':
                # name_dic 업데이트
                if name_dic :
                    last_name=list(name_dic.keys())[-1]
                    x=str(last_name).split('_')[-1]
                    name = f'unknown_{int(x)+1}'
                else :
                    name = f'unknown_{len(name_dic)+1}'
                name_dic[name]=1
            # 아는 extra가 있는 경우
            else :
                name_dic[name]+=1
        # df 업데이트
        df.loc[len(df)] = [pic_name[:-4],x_left, y_left, x_right, y_right,name]
        save_crop_img(df,img_cropped,name)

        # check_to_value(img_cropped, name_dic, df)

        print("Training KNN classifier...")
        classifier = train(model,device, mtcnn,"actor", model_save_path="trained_knn_model.clf", n_neighbors=2)
        print("Training complete!")

    return df


def first_use(input_dir:str, model, device, mtcnn, name_dic:dict, df):
    '''
        {input_dir} - 업로드할 이미지가 있는 폴더
                    - ex.) "./harrypotter/"
    '''
    verbose = False
    cycle=0
    for idx, pic_name in enumerate(os.listdir(input_dir)):
        orig_image = Image.open(input_dir + pic_name)

        # 첫번째로 입력된 사진인 경우
        if cycle == 0:
            image = np.array(orig_image)
            face_locations = mtcnn_detect(image, mtcnn)

            for i, face_location in enumerate(face_locations):
                x_left, y_left, x_right, y_right = face_location
                no_name = "unknown_{}".format(i)

                # name_dic 업데이트
                name_dic[no_name]=1

                # df 업데이트
                df.loc[len(df)] = [pic_name[:-4],x_left, y_left, x_right, y_right,no_name]
                img_cropped = orig_image.crop((x_left, y_left, x_right, y_right))
                plt.title(no_name)
                plt.imshow(img_cropped)

                # 사진에서 얼굴이 없는 경우(또는 인식못한 경우)
                if len(face_locations) != 1:
                    if verbose:
                        print("Image {} not suitable for training: {}".format(pic_name, "Didn't find a face" if len(face_locations) < 1 else "Found more than one face"))
                plt.show()
                
                # 라벨 요청
                answer = input(f"Do you want to specify a label  for {no_name}?[y/n] : ")
                if answer == 'y':
                    name = input(f"Who is {no_name} : ")
                    df = df.replace(no_name,name)  # df 업데이트
                    del(name_dic[no_name])  # name_dic 업데이트
                else :
                    name = no_name
                save_crop_img(df,img_cropped,name)
            cycle+=1

            # Train the KNN classifier and save it to disk
            print("Training KNN classifier...")
            classifier = train(model, device, mtcnn, "actor", model_save_path="trained_knn_model.clf", n_neighbors=2)
            print("Training complete!")
        
        # 첫번째 사진을 제외한 사진들에 대해서 진행
        else:
            inference(model,device,mtcnn, input_dir, pic_name, orig_image, name_dic, df)
        print(idx)
    return df


def second_use(input_dir:str, model, device, mtcnn, name_dic:dict, df):
    for idx, pic_name in enumerate(os.listdir(input_dir)):
        if pic_name in df['photo_ID'].unique():
            continue
        orig_image = Image.open(input_dir + pic_name)
        inference(model, device, mtcnn, input_dir, pic_name, orig_image, name_dic, df)
        print(idx)
    return df

