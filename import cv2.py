import cv2

def main():
    cap = cv2.VideoCapture(0)  # Open the live camera feed (0 represents the default camera)

    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        
        if not ret:
            break
        
        process_frame(frame)
        
        cv2.imshow('Live Camera Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (13, 13), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    dilate = cv2.dilate(dilate, vertical_kernel, iterations=2)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 20000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 3)

if __name__ == "__main__":
    main()
