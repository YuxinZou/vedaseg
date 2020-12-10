import os

import cv2
import numpy as np
import json

from vedaseg.datasets.base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class ActivityRawFrameDataset(BaseDataset):
    CLASSES = ('Applying sunscreen', 'Arm wrestling', 'Assembling bicycle', 'BMX', 'Baking cookies', 'Baton twirling', 'Beach soccer', 'Beer pong', 'Blow-drying hair', 'Blowing leaves', 'Playing ten pins', 'Braiding hair', 'Building sandcastles', 'Bullfighting', 'Calf roping', 'Camel ride', 'Canoeing', 'Capoeira', 'Carving jack-o-lanterns', 'Changing car wheel', 'Cleaning sink', 'Clipping cat claws', 'Croquet', 'Curling', 'Cutting the grass', 'Decorating the Christmas tree', 'Disc dog', 'Doing a powerbomb', 'Doing crunches', 'Drum corps', 'Elliptical trainer', 'Doing fencing', 'Fixing the roof', 'Fun sliding down', 'Futsal', 'Gargling mouthwash', 'Grooming dog', 'Hand car wash', 'Hanging wallpaper', 'Having an ice cream', 'Hitting a pinata', 'Hula hoop', 'Hurling', 'Ice fishing', 'Installing carpet', 'Kite flying', 'Kneeling', 'Knitting', 'Laying tile', 'Longboarding', 'Making a cake', 'Making a lemonade', 'Making an omelette', 'Mooping floor', 'Painting fence', 'Painting furniture', 'Peeling potatoes', 'Plastering', 'Playing beach volleyball', 'Playing blackjack', 'Playing congas', 'Playing drums', 'Playing ice hockey', 'Playing pool', 'Playing rubik cube', 'Powerbocking', 'Putting in contact lenses', 'Putting on shoes', 'Rafting', 'Raking leaves', 'Removing ice from car', 'Riding bumper cars', 'River tubing', 'Rock-paper-scissors', 'Rollerblading', 'Roof shingle removal', 'Rope skipping', 'Running a marathon', 'Scuba diving', 'Sharpening knives', 'Shuffleboard', 'Skiing', 'Slacklining', 'Snow tubing', 'Snowboarding', 'Spread mulch', 'Sumo', 'Surfing', 'Swimming', 'Swinging at the playground', 'Table soccer', 'Throwing darts', 'Trimming branches or hedges', 'Tug of war', 'Using the monkey bar', 'Using the rowing machine', 'Wakeboarding', 'Waterskiing', 'Waxing skis', 'Welding', 'Drinking coffee', 'Zumba', 'Doing kickboxing', 'Doing karate', 'Tango', 'Putting on makeup', 'High jump', 'Playing bagpipes', 'Cheerleading', 'Wrapping presents', 'Cricket', 'Clean and jerk', 'Preparing pasta', 'Bathing dog', 'Discus throw', 'Playing field hockey', 'Grooming horse', 'Preparing salad', 'Playing harmonica', 'Playing saxophone', 'Chopping wood', 'Washing face', 'Using the pommel horse', 'Javelin throw', 'Spinning', 'Ping-pong', 'Making a sandwich', 'Brushing hair', 'Playing guitarra', 'Doing step aerobics', 'Drinking beer', 'Playing polo', 'Snatch', 'Paintball', 'Long jump', 'Cleaning windows', 'Brushing teeth', 'Playing flauta', 'Tennis serve with ball bouncing', 'Bungee jumping', 'Triple jump', 'Horseback riding', 'Layup drill in basketball', 'Vacuuming floor', 'Cleaning shoes', 'Doing nails', 'Shot put', 'Fixing bicycle', 'Washing hands', 'Ironing clothes', 'Using the balance beam', 'Shoveling snow', 'Tumbling', 'Using parallel bars', 'Getting a tattoo', 'Rock climbing', 'Smoking hookah', 'Shaving', 'Getting a piercing', 'Springboard diving', 'Playing squash', 'Playing piano', 'Dodgeball', 'Smoking a cigarette', 'Sailing', 'Getting a haircut', 'Playing lacrosse', 'Cumbia', 'Tai chi', 'Painting', 'Mowing the lawn', 'Shaving legs', 'Walking the dog', 'Hammer throw', 'Skateboarding', 'Polishing shoes', 'Ballet', 'Hand washing clothes', 'Plataform diving', 'Playing violin', 'Breakdancing', 'Windsurfing', 'Hopscotch', 'Doing motocross', 'Mixing drinks', 'Starting a campfire', 'Belly dance', 'Removing curlers', 'Archery', 'Volleyball', 'Playing water polo', 'Playing racquetball', 'Kayaking', 'Polishing forniture', 'Playing kickball', 'Using uneven bars', 'Washing dishes', 'Pole vault', 'Playing accordion', 'Playing badminton')

    def __init__(self,
                 root,
                 ann_file,
                 img_prefix,
                 nclasses=20,
                 fps=10,
                 transform=None,
                 multi_label=True,
                 mask_value=255,
                 ):
        super().__init__()
        self.root = root
        self.data = json.load(open(os.path.join(self.root, ann_file), 'r'))
        self.multi_label = multi_label
        self.transform = transform
        self.img_prefix = img_prefix
        self.mask_value = mask_value
        self.nclasses = nclasses
        self.fps = fps
        self.video_names = list(self.data.keys())
        if self.root is not None:
            self.img_prefix = os.path.join(self.root, self.img_prefix)

        self.cap = cv2.VideoCapture()


    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                True).
        """
        while True:
            data = self.prepare(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def prepare(self, item):
        frame_dir = os.path.join(self.img_prefix, self.video_names[item])
        gt = self.data[self.video_names[item]]
        fnames = sorted(os.listdir(frame_dir))
        fnames = [os.path.join(frame_dir, img) for img in fnames]

        labels = []
        segments = []

        for anno in gt['annotations']:
            segment = [int(i * self.fps) for i in anno['segment']]
            if segment[1] - segment[0] <= 0:
                continue

            segments.append(segment)
            labels.append(self.CLASSES.index(anno['label']))

        if len(segments) == 0:
            return None

        # mask shape C*T
        data = dict(image=fnames, duration=len(fnames),
                    labels=np.array(labels), segments=np.array(segments))
        image, mask = self.process(data)

        return image.float(), mask.long()

    def __len__(self):
        return len(self.video_names)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(len(self.video_names))[0]
        return np.random.choice(pool)
