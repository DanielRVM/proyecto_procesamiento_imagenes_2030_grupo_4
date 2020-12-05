# Importamos librerías
import cv2
import numpy as np
 

cap = cv2.VideoCapture(0) #cptura video de la camára 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480);
#pixels
heightImg = 640 
widthImg  = 480


#Función para ordenar

def reorder(myPoints):
 
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
 
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
 
    return myPointsNew

#Función para mostrar los resultados en simultáneo

def stackImages(imgArray,scale,lables=[]):
    #numero de filas
    rows = len(imgArray)
    #numero de columnas 
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        #for para recorrer filas y columnas
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale) #resize
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
    return ver
 
#Funcíon para encontrar el contorno de mayor área
 
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

#Función para dibujar el perímetro del contorno más grande

def drawRectangle(img,biggest,thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 255), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 255), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 255), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 255), thickness)
 
    return img

 
def nothing(x):
    pass 

#Funcíon para crear una ventana con barras de seguimiento de los valores de los threshold utilizados en la función Canny
 
def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200,255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)
 
 
#Función utilizada para saber la posición actual de la barra de seguimiento de los threshold

def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1,Threshold2
    return src

initializeTrackbars()
count=0

while True:
 
    ret, img = cap.read()
    if img is not None:
        img = cv2.resize(img, (widthImg, heightImg)) #Escalamiento de la imagen
        imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) #cra imagen blanca del tamaño del frame
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #copia en escala de grises
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) #aplica desenfoque gaussiano
        threshold=valTrackbars() #valores a barras de seguimiento
        imgThreshold = cv2.Canny(imgBlur,threshold[0],threshold[1]) # aplica canny
        kernel = np.ones((5, 5)) #kernel de 5x5
        imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # aplica dilatación
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # luego hacemos erosión
     
        ## para encontrar contornos
        imgContours = img.copy()
        imgBigContour = img.copy() 
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # función que encuentra contornos
        #cv2.drawContours(imgContours, contours, -1, (255, 0, 0), 5) # dibuja contornos encontrados en el punto anterior
     
     
        
        biggest, maxArea = biggestContour(contours) # encuentra el más grande 
        if biggest.size != 0:
            biggest=reorder(biggest) #ordena los valores entregados por los la función biggest contours
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 255), 8) #dibuja contorno encontrado en el punto anterior
            imgBigContour = drawRectangle(imgBigContour,biggest,2)
            #preparando los puntos para deformación
            pts1 = np.float32(biggest) 
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) 
            matrix = cv2.getPerspectiveTransform(pts1, pts2) #aplica transformacion de perspectiva
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
     
            # Umbral adaptativo
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
            imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
            imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)
     
            # Image Array for Display
            imageArray = ([img,imgGray,imgThreshold,imgContours],
                          [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])
     
        else:
            imageArray = ([img,imgGray,imgThreshold,imgContours],
                          [imgBlank, imgBlank, imgBlank, imgBlank])
     
        # Nombres de cada resultado mostrado en pantalla
        
        lables = [["Original","Escala de grises","Umbralizacion","Contornos"],
                  ["Mayor contorno","Trans. de perspectiva","Trans. en grises","Umbral adaptativo"]]
     
        stackedImage = stackImages(imageArray,0.5,lables)
        cv2.imshow("Resultado",stackedImage)
        
        # Guarda imagen escaneada presionando 's'
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("scan"+str(count)+".jpg",imgWarpColored)
            cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                          (1100, 350), (0, 255, 0), cv2.FILLED)
            cv2.putText(stackedImage, "Imagen escaneada guardada", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                        cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.imshow('Resultado', stackedImage)
            cv2.waitKey(300)
            count += 1
    else:
        print(" Saliendo...")
        break
cv2.destroyAllWindows()
cv2.waitKey(1)
        