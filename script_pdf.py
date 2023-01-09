import cv2
import numpy as np
import os
import pytesseract
import pandas as pd
from pdf2image import convert_from_path

#Convert given document to images and save them to a folder named "data"
images = convert_from_path('Label.pdf')
for i in range(len(images)):
    images[i].save('data/page'+ str(i) +'.jpg', 'JPEG')

# Get detected symbols and their corresponding scores
def get_scores(edged):
    scores = list()
    for temp_name in os.listdir("symbols"):
        template = cv2.imread(os.path.join("./symbols", temp_name))
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (w, h) = template.shape[:2]
        found = None
        for h_red in np.linspace(0.9, 1.6, 35)[::-1]:
            for w_red in np.linspace(0.5, 0.8, 20)[::-1]:
                dim = (int(h*h_red), int(w * w_red))
                res = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)
                result = cv2.matchTemplate(edged, res, cv2.TM_CCORR_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc)
        scores.append(found[0])
    scores = np.array(scores)
    args = np.squeeze(np.argwhere(scores>0.37)) + 1
    if(type(args)==np.int64):
        temp = args.copy()
        args = list()
        args.append(temp)
    return scores, args

# Process detections to eliminate false detections and convert them to string
def process(args, scores):
    ac = None
    l = list()
    seq = None
    if 1 in args and 6 in args and 5 not in args:
        mx = max(scores[0], scores[5])
        ac = np.squeeze(np.argwhere(scores == mx)) + 1
        for a in args:
            if(a ==1 or a ==6 and a != ac ):
                continue
            else:
                l.append(str(a))
        seq = "".join(l)
    elif 1 in args and 5 in args and 6 in args:
        one = False
        five = False
        six = False
        if(scores[0]>0.56):
            one = True
        if(scores[4]>0.4):
            five = True
        if(scores[5]>0.38):
            six = True
        for a in args:
            if a ==1 or a ==5 or a ==6:
                if(a ==1 and one ):
                    l.append(str(a))
                if(a ==5 and five ):
                    l.append(str(a))
                if(a ==6 and six ):
                    l.append(str(a))
            else:
                l.append(str(a))
        #print(type(l[0]))
        seq = "".join(l)
    else:
        for a in args:
            l.append(str(a))
        seq = "".join(l)
    return seq

# Extract text data
def get_details(image):
    data = pytesseract.image_to_string(image).split("\n")
    line = ' '.join(data).split()
    name = list()
    ref = None
    qty = None
    lot = None
    for i, word in enumerate(line):
        if(word == "Name:"):
            c = i
            while(line[c+1] != "REF"):
                name.append(line[c+1])
                c += 1
            name = " ".join(name)
        if(word == "REF"):
            ref = line[i+1]
        if(word == "Qty:"):
            qty = line[i+1]
        if(word == "LOT:"):
            lot = line[i+1]
    return name, ref, qty, lot

# Putting things together and generate output csv file
master_list = list()
for im_name in os.listdir("data"):
    details = list()
    image = cv2.imread(os.path.join("data", im_name))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 200)
    scores, args = get_scores(edged)
    sym_labels = process(args, scores)
    name, ref, _, lot = get_details(os.path.join("data", im_name))
    _, _, qty, _ = get_details(edged)
    details.append(name)
    details.append(ref)
    details.append(lot)
    details.append(qty)
    details.append(sym_labels)
    master_list.append(details)

df = pd.DataFrame(master_list, columns = ['Device Name', 'REF', 'LOT', 'Qty', 'Symbols'])
df.to_csv("submission_pdf.csv")