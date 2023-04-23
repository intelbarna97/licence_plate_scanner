import cv2
import imutils
import pytesseract
import numpy as np
from skimage.segmentation import clear_border

class PlateDetector:
    def __init__(self, image):
        self.image = image
        pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

    def detect(self):
        self.image = imutils.resize(
            self.image, width=1000)  # resize image to 1000
        # cv2.imshow('original video', self.image)
        original_image = self.image.copy()
        gray_image = cv2.cvtColor(
            self.image, cv2.COLOR_BGR2GRAY)  # convert to gray
        gray_image = cv2.bilateralFilter(
            gray_image, 3, 6, 6)  # smoothen image 3 6 6
        gray_image = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        cv2.imshow('gray', gray_image)
        # edged = cv2.Canny(gray_image, 30, 200)  # edge detection
        # cv2.imshow('edged', edged)
        cnts, _ = cv2.findContours(  # find contours
            gray_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[
            10:100]  # sort contours
        cv2.drawContours(original_image, cnts, -1, (0, 255, 0), 3)
        cv2.imshow("Top contours", original_image)
        for c in cnts:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)  # 0.01
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(c)
                new_img = self.image[y:y+h, x:x+w]
                aspect_ratio = float(new_img.shape[1])/float(new_img.shape[0])
                if aspect_ratio > 4.5 and aspect_ratio < 5.0:
                    cv2.imshow("possible plate", new_img)
                    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                    print(pytesseract.image_to_string(new_img, lang='eng'))


class PyImageSearchANPR:
    def __init__(self, minAR=4, maxAR=5, debug=False):
        # store the minimum and maximum rectangular aspect ratio
        # values along with whether or not we are in debug mode
        self.minAR = minAR
        self.maxAR = maxAR
        self.debug = debug

    def debug_imshow(self, title, image, waitKey=False):
        # check to see if we are in debug mode, and if so, show the
        # image with the supplied title
        if self.debug:
            cv2.imshow(title, image)
            # check to see if we should wait for a keypress
            if waitKey:
                cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        # perform a blackhat morphological operation that will allow
        # us to reveal dark regions (i.e., text) on light backgrounds
        # (i.e., the license plate itself)
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.debug_imshow("Blackhat", blackhat)
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Light Regions", light)
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
                          dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debug_imshow("Scharr", gradX)
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debug_imshow("Grad Erode/Dilate", thresh)
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        # return the list of contours
        return cnts

    def locate_license_plate(self, gray, candidates,
                             clearBorder=False):
        # initialize the license plate contour and ROI
        lpCnt = None
        roi = None
        # loop over the license plate candidate contours
        for c in candidates:
            # compute the bounding box of the contour and then use
            # the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
        # check to see if the aspect ratio is rectangular
            if ar >= self.minAR and ar <= self.maxAR:
                # store the license plate contour and extract the
                # license plate from the grayscale image and then
                # threshold it
                lpCnt = c
                licensePlate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                if clearBorder:
                    roi = clear_border(roi)
                self.debug_imshow("License Plate", licensePlate)
                self.debug_imshow("ROI", roi)
                break
        return (roi, lpCnt)

    def find_and_ocr(self, image, psm=7, clearBorder=False):
        # initialize the license plate text
        lpText = None
        pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'
        # convert the input image to grayscale, locate all candidate
        # license plate regions in the image, and then process the
        # candidates, leaving us with the *actual* license plate
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        (lp, lpCnt) = self.locate_license_plate(gray, candidates,
                                                clearBorder=clearBorder)
        # only OCR the license plate if the license plate ROI is not
        # empty
        if lp is not None:
            lpText = pytesseract.image_to_string(lp, lang="eng",)
            self.debug_imshow("License Plate", lp)
            print(lpText)
        # return a 2-tuple of the OCR'd license plate text along with
        # the contour associated with the license plate region
        return (lpText, lpCnt)


if __name__ == "__main__":

    cap = cv2.VideoCapture('videoplayback.mp4')
    while (cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            # detector = PyImageSearchANPR(debug=True)
            detector = PlateDetector(img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # cv2.waitKey(0)
            detector.detect()
            # detector.find_and_ocr(img)

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
