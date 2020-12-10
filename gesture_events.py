from datetime import datetime


class UpEvent(Exception):
    pass


class DownEvent(Exception):
    pass


class DragEvent(Exception):
    pass


class DoubleClickEvent(Exception):
    pass


class LeftClickEvent(Exception):
    pass


class RightClickEvent(Exception):
    pass


event_queue = ['none']
last_event_time = datetime.now()
gesture_mapper = {'up': UpEvent, 'down': DownEvent, 'vertical': DragEvent, 'horizontal': DoubleClickEvent,
                  'left': LeftClickEvent, 'right': RightClickEvent, 'fist': Exception}


def process_gesture(gesture):
    limit = 15
    delay = 1
    global event_queue, last_event_time
    event_queue.append(gesture)
    event_queue = event_queue[-limit:]
    if len(event_queue) < limit:
        return

    if len(set(event_queue)) != 1:
        return

    if event_queue[-1] == gesture and \
            (datetime.now() - last_event_time).total_seconds() < delay:
        return

    event_queue = []
    last_event_time = datetime.now()
    raise gesture_mapper[gesture]
