import cv2

image = cv2.imread('Images_upload\\image.jpg')
reShape = cv2.resize(image, (545, 427))

alpha = 1
beta = 0

alter = cv2.convertScaleAbs(image, alpha  = alpha, beta = beta)

cv2.imshow('adjusted', alter[:,:,::])
cv2.waitKey()
cv2.destroyAllWindows()