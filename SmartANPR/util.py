import string
from collections import defaultdict

import cv2
import numpy as np
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Minimum plate-crop height (px) below which we upscale before OCR.
# Plates are small, high-detail objects; EasyOCR benefits from more pixels.
_MIN_OCR_HEIGHT = 64

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def _upscale_if_small(img):
    """Upscale a crop if it is smaller than _MIN_OCR_HEIGHT, preserving aspect."""
    if img is None or img.size == 0:
        return img
    h = img.shape[0]
    if h >= _MIN_OCR_HEIGHT:
        return img
    scale = _MIN_OCR_HEIGHT / float(h)
    new_w = max(1, int(round(img.shape[1] * scale)))
    return cv2.resize(img, (new_w, _MIN_OCR_HEIGHT), interpolation=cv2.INTER_CUBIC)


def enhance_plate_crop(license_plate_crop):
    """
    Conservative, lightweight enhancement of a license-plate crop before OCR.

    Steps (all cheap, no training): optional upscaling of small crops,
    grayscale, CLAHE contrast equalization, and Otsu thresholding. This is
    offered as an ALTERNATIVE attempt in read_license_plate() - the original
    crop is always tried too, so this can only add a second chance and never
    makes an already-good crop worse.

    Args:
        license_plate_crop (np.ndarray): Cropped plate image (BGR, grayscale,
            or already-thresholded).

    Returns:
        np.ndarray or None: Enhanced single-channel image, or None if the input
            is empty.
    """
    if license_plate_crop is None or getattr(license_plate_crop, "size", 0) == 0:
        return None

    img = _upscale_if_small(license_plate_crop)

    # Reduce to a single channel regardless of input format.
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Contrast enhancement that is robust to uneven lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Otsu picks the threshold from the data instead of a fixed value.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Tries the crop as received first (baseline behavior), then a conservatively
    enhanced variant, and returns the highest-confidence result that complies
    with the plate format. Trying the original first means already-good crops
    are never degraded by the extra preprocessing.

    Args:
        license_plate_crop (np.ndarray): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    candidates = [license_plate_crop]
    enhanced = enhance_plate_crop(license_plate_crop)
    if enhanced is not None:
        candidates.append(enhanced)

    best_text, best_score = None, None
    for variant in candidates:
        if variant is None or getattr(variant, "size", 0) == 0:
            continue
        detections = reader.readtext(variant)
        for detection in detections:
            bbox, text, score = detection

            text = text.upper().replace(' ', '')

            if license_complies_format(text) and (best_score is None or score > best_score):
                best_text, best_score = format_license(text), score

    return best_text, best_score


def aggregate_plates_by_car(results):
    """
    Aggregate per-frame OCR reads into one final plate string per car_id.

    Combines consistency (how many frames agree on a string) with OCR
    confidence, so a single low-confidence read cannot overwrite a consistent
    higher-confidence result.

    Selection rule (in order):
      1. Highest vote count (frequency / consistency across frames).
      2. Tie-break: highest summed confidence.
      3. Tie-break: highest single-frame confidence.

    Ranking frequency first means a plate string that is read consistently
    across several frames wins over a lone read, even a lone high-confidence
    one; among equally-consistent candidates, confidence decides.

    Args:
        results (dict): The same {frame_nmr: {car_id: {...}}} structure that
            main.py builds and write_csv() consumes.

    Returns:
        dict: {car_id: {'license_plate': str,
                        'score': float,        # summed confidence (support)
                        'best_frame_score': float,
                        'num_votes': int,      # frames agreeing on the winner
                        'num_reads': int}}     # total valid reads for the car
    """
    # car_id -> plate_text -> list of scores
    votes = defaultdict(lambda: defaultdict(list))

    for frame_nmr in results:
        for car_id in results[frame_nmr]:
            entry = results[frame_nmr][car_id]
            if 'license_plate' not in entry:
                continue
            lp = entry['license_plate']
            text = lp.get('text')
            score = lp.get('text_score')
            if text is None or score is None:
                continue
            votes[car_id][text].append(float(score))

    aggregated = {}
    for car_id, plate_scores in votes.items():
        best_plate = None
        best_key = None       # (num_votes, summed_confidence, single_best)
        best_support = 0.0
        best_single = 0.0
        best_votes = 0
        total_reads = 0
        for text, scores in plate_scores.items():
            total_reads += len(scores)
            support = sum(scores)
            single = max(scores)
            key = (len(scores), support, single)  # frequency first, then confidence
            if best_key is None or key > best_key:
                best_key = key
                best_plate = text
                best_support = support
                best_single = single
                best_votes = len(scores)

        if best_plate is not None:
            aggregated[car_id] = {
                'license_plate': best_plate,
                'score': best_support,
                'best_frame_score': best_single,
                'num_votes': best_votes,
                'num_reads': total_reads,
            }

    return aggregated


def write_aggregated_csv(aggregated, output_path):
    """
    Write one aggregated (final) plate row per car_id.

    This is an ADDITIONAL output; it does not alter the per-frame test.csv.

    Args:
        aggregated (dict): Output of aggregate_plates_by_car().
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{}\n'.format(
            'car_id', 'license_plate', 'aggregate_score',
            'best_frame_score', 'num_votes', 'num_reads'))
        for car_id in sorted(aggregated.keys()):
            a = aggregated[car_id]
            f.write('{},{},{},{},{},{}\n'.format(
                car_id,
                a['license_plate'],
                a['score'],
                a['best_frame_score'],
                a['num_votes'],
                a['num_reads']))


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
