import cv2
import numpy as np
import imutils
from imutils import perspective
import easyocr
import string
import os


class PlateProcessing():
    def __init__(self):
        self.rectKern = rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        pass

    def ratioCheck(self, width, height):
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        if ratio < 3 or ratio > 6:
            return False
        return True

    def clean_plate(self, plate):
        image_copy = plate.copy()
        gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.bilateralFilter(gray_img, 13, 15, 15)
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.dilate(thresh, np.ones((2, 2), dtype="uint8"))
        # cv2.imshow("thresh", cv2.resize(thresh, dsize=None, fx=5, fy=5))
        num_contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(image_copy, num_contours, -1, (0, 255, 0), 1)
        # cv2.imshow("image_copy", cv2.resize(image_copy, dsize=None, fx=5, fy=5))
        if num_contours:
            contour_area = [cv2.contourArea(c) for c in num_contours]
            max_cntr_index = np.argmax(contour_area)
            max_cnt = num_contours[max_cntr_index]
            max_cntArea = contour_area[max_cntr_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            if not self.ratioCheck(w, h):
                return plate, None, None
            final_img = thresh[y:y + h, x:x + w]
            rect = cv2.minAreaRect(max_cnt)
            box = np.int0(cv2.boxPoints(rect))
            return final_img, [x, y, w, h], box
        else:
            return plate, None, None

    def morphology_operation(self, gray, rectKern):
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        # cv2.imshow("Blackhat", imutils.resize(blackhat, width=400))
        # cv2.imshow("Closing operation", imutils.resize(light, width=400))
        light = cv2.threshold(light, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # light = cv2.dilate()
        return imutils.resize(light, width=400)
        # cv2.imshow("Light Regions", imutils.resize(light, width=400))
        # cv2.waitKey()
        # return [blackhat, light]

    def CCA_contour(self, image):
        H, W = image.shape[:2]
        bboxes = []
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 800:
                continue
            if H / h > 10:
                continue
            if w > W / 2:
                continue
            # if (y <= 10 or y + h > H - 10) and (x < 10 or x + w > W - 10):
            #     continue
            bboxes.append([x, y, w, h])
        bboxes_new = []
        for i, bbox in enumerate(sorted(bboxes, key=lambda a: a[0])):
            x, y, w, h = bbox
            if self.is_overlap(x, y, w, h, sorted(bboxes, key=lambda a: a[0])[:i]):
                continue
            bboxes_new.append([x, y, w, h])
        return bboxes_new

    def is_lp_square(self, image):
        H, W = image.shape[:2]
        if W / H > 2.5:
            return False
        return True

    def is_overlap(self, x, y, w, h, bboxes):
        x_cen = x + w // 2
        y_cen = y + h // 2
        for x1, y1, w1, h1 in bboxes:
            if (x1 < x_cen < x1 + w1) and (y1 < y_cen < y1 + h1):
                return True

    def bboxes_square_lp(self, image):
        bboxes = self.CCA_contour(image)
        y_mean = sum([i[1] for i in bboxes]) / len(bboxes)
        line1 = []
        line2 = []
        for bbox in bboxes:
            x, y, w, h = bbox
            if y < y_mean:
                line1.append([x, y, w, h])
            else:
                line2.append([x, y, w, h])
        line1 = sorted(line1, key=lambda a: a[0])
        line2 = sorted(line2, key=lambda a: a[0])
        bboxes_LP = line1 + line2
        return bboxes_LP

    def bboxes_rec_lp(self, image):
        bboxes = self.CCA_contour(image)
        return sorted(bboxes, key=lambda a: a[0])

    def add_extend(self, image, filters="black", width=10, height=10):
        H, W = image.shape[:2]
        if filters == "black":
            blank_image = np.zeros((H + height * 2, W + width * 2), np.uint8)
        elif filters == "white":
            blank_image = np.ones((H + height * 2, W + width * 2), np.uint8) * 255
        blank_image[height:H + height, width:W + width] = image
        return blank_image

    def process_bboxes(self, bboxes):
        bboxes_new = []
        w_mean = sorted(bboxes, key=lambda x: x[2])[2][2]
        for x, y, w, h in bboxes:
            if w / w_mean > 1.7:
                bboxes_new.append([x, y, w // 2, h])
                bboxes_new.append([x + w // 2, y, w // 2, h])
            else:
                bboxes_new.append([x, y, w, h])
        return bboxes_new

    def lp_detection(self, img):
        plate, bbox, box = self.clean_plate(img)
        if bbox is not None:
            x, y, w, h = bbox
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plate_warp = perspective.four_point_transform(img, box)
            image = imutils.resize(plate_warp, width=400)
            image = cv2.bilateralFilter(image, 3, 105, 105)
            # cv2.imshow("Bilateral Filter", image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            light = self.morphology_operation(gray, self.rectKern)
            light = ~light
            light = cv2.erode(light, np.ones((2, 2), dtype="uint8"), iterations=2)
            if self.is_lp_square(light):
                bboxes = self.bboxes_square_lp(light)
            else:
                bboxes = self.bboxes_rec_lp(light)
            bboxes = self.process_bboxes(bboxes)
            return bboxes, light

        else:
            return None, None


class PlateRecognition(PlateProcessing):
    def __init__(self):
        PlateProcessing.__init__(self)
        self.reader = easyocr.Reader(lang_list=["en"])
        self.allowlist = string.ascii_uppercase + string.digits
        pass

    def image_to_text(self, image):
        return self.reader.readtext(image, allowlist=self.allowlist, decoder="greedy", min_size=5,
                                    text_threshold=0.4, low_text=0.2, link_threshold=0.2, mag_ratio=3, detail=0)[0]

    def lp_recognition(self, image):
        bboxes, light = self.lp_detection(image)
        if bboxes is None:
            return "", image
        lp_text = ""
        for x, y, w, h in bboxes:
            crop = light[y:y + h, x:x + w]
            crop_extend = self.add_extend(crop)
            crop = cv2.resize(crop_extend, (28, 28), interpolation=cv2.INTER_AREA)
            lp_text += self.image_to_text(crop)
        return lp_text, light


if __name__ == "__main__":
    path = r"D:\self_project\yolor\ANPR\no_warp"
    plate_recognition = PlateRecognition()
    for imageName in os.listdir(path):
        img = cv2.imread(f"{path}/{imageName}")
        lp_text, light = plate_recognition.lp_recognition(img)
        print(lp_text)
        cv2.imshow("Light", light)
        cv2.waitKey()
