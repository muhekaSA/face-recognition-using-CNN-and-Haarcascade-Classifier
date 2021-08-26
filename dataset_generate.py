import  cv2
def generate_dataset():
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    def face_cropped(img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_gray_detect = face_classifier.detectMultiScale(gray_img, 1.3, 5)
        
        if faces_gray_detect is ():
            return None
        for (x,y,w,h) in faces_gray_detect:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
    
    vid_realTime0 = cv2.VideoCapture(0)
    vid_realTime1 = cv2.VideoCapture(1)
    img_ida = 0
    img_idb = 0
    
    while True:
        
        ret0, frame0 = vid_realTime0.read()
        ret1, frame1 = vid_realTime1.read()
        
        if face_cropped(frame0) is not None:
            img_ida+=1
            face = cv2.resize(face_cropped(frame0), (200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #source_output_img = "Face_Dataset/"+"eka."+str(img_ida)+'.jpg'
            source_output_img = "Images for visualization/"+str(img_ida)+'.jpg'
            cv2.imwrite(source_output_img, face)
            cv2.putText(face, str(img_ida), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
            
            cv2.imshow("Cropped_Face", face)
            if cv2.waitKey(1)==13 or int(img_ida)==20:
                break
        
                
            
        if face_cropped(frame1) is not None:
            img_idb+=1
            face2 = cv2.resize(face_cropped(frame1), (200,200))
        
            source_output_img2 = "RGB/"+"ekaRGB."+str(img_idb)+'.jpg'
            cv2.imwrite(source_output_img2, face2)
            cv2.putText(face2, str(img_idb), (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2 )
            
            cv2.imshow("Cropped_Face2", face2)
            if cv2.waitKey(1)==13 or int(img_idb)==1000:
                break
       
    vid_realTime0.release()
    vid_realTime1.release()
    
    cv2.destroyAllWindows()
    print("Collecting is Done... !!!")              
generate_dataset()