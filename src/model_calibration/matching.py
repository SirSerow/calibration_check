"""Use COCOeval directly, retaining its ignore and duplicate decisions."""
import contextlib
import io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(annotation_path, predictions, image_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation_path))
        flat, identities = [], {}
        for image in predictions:
            for d in image['detections']:
                x1, y1, x2, y2 = d['box']
                flat.append(dict(image_id=image['image_id'], category_id=d['category_id'],
                                 # Malformed native boxes are retained as zero-area false positives.
                                 bbox=[x1,y1,max(0,x2-x1),max(0,y2-y1)], score=d['score']))
                identities[len(flat)] = d['detection_id']
        if flat:
            dt = gt.loadRes(flat)
        else:
            dt = COCO()
            dt.dataset = {'images': gt.dataset['images'], 'categories': gt.dataset['categories'], 'annotations': []}
            dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = list(map(int, image_ids))
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    rows = {0.5: [], 0.75: []}
    for entry in ev.evalImgs:
        if entry is None or entry['aRng'] != ev.params.areaRng[0]:
            continue
        for iou in rows:
            t = int(np.flatnonzero(np.isclose(ev.params.iouThrs, iou))[0])
            for j, dt_id in enumerate(entry['dtIds']):
                ann = dt.anns[dt_id]
                rows[iou].append(dict(image_id=int(entry['image_id']), detection_id=identities[dt_id],
                                      category_id=int(entry['category_id']), score=ann['score'],
                                      correct=int(entry['dtMatches'][t,j] > 0),
                                      ignored=bool(entry['dtIgnore'][t,j])))
    return rows, dict(map=float(ev.stats[0]), map50=float(ev.stats[1]), map75=float(ev.stats[2]),
                      recall100=float(ev.stats[8]))
