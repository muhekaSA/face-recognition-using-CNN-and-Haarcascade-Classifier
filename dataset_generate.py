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
    
    vid_realTime = cv2.VideoCapture(0)
    img_id = 0
    
    while True:
        ret, frame = vid_realTime.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #file_name_path = "data/"+"Ishwar."+str(img_id)+".jpg"
            source_output_img = "Face_Dataset/"+"eka."+str(img_id)+'.jpg'
            cv2.imwrite(source_output_img, face)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
            
            cv2.imshow("Cropped_Face", face)
            if cv2.waitKey(1)==13 or int(img_id)==1000:
                break
                
    vid_realTime.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed !!!")
generate_dataset()