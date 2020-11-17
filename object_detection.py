from imageai.Detection import ObjectDetection
import os
import sys

"""
General object detection using RetinaNet using pretrained Object Detector
reference : ImageAi opensource library
"""
class object_detector:
    """
    class for object detector, can be easily modified to suit multiple detectors
    """
    def __init__(self, model, validation, output):
        """
        Init
        :param model: path to folder where model is stored
        :param validation: path to images folder to test detector
        :param output: path to output folder to store images
        """
        self.path_of_model = model
        self.validation_path = validation
        self.output_path = output

    def check_paths(self, path):
        """
        Check validity of input
        :param path: path of folder under consideration
        :return: path of folder if correct else close the program
        """
        if path is "model" and os.path.exists(self.path_of_model):
            if os.path.exists(self.path_of_model + "\\resnet50_coco_best_v2.0.1.h5"):
                return os.path.abspath(self.path_of_model + "\\resnet50_coco_best_v2.0.1.h5")
            else:
                print("Please ensure model file is present in model folder")
        elif path is "validation" and os.path.exists(self.validation_path):
            return [image for image in os.listdir(os.path.abspath(self.validation_path))], os.path.abspath(
                self.validation_path)
        elif path is "output" and os.path.exists(self.output_path):
            return os.path.abspath(self.output_path)
        else:
            print("Please check path for: " + path)
            sys.exit(-1)

    def detect(self):
        """
        method to create and deploy object detector
        :return: None
        """
        classifier = ObjectDetection()
        classifier.setModelTypeAsRetinaNet()
        classifier.setModelPath(self.check_paths("model"))
        classifier.loadModel()
        input_images, input_path = self.check_paths("validation")[0], self.check_paths("validation")[1]
        if len(input_images) ==0 :
            print("Please check input images folder")
            sys.exit(-1)
        else:
            output_path = self.check_paths("output")
            for image in input_images:
                objects_detected = classifier.detectObjectsFromImage(input_path + "\\" + image,
                                                                 output_path + "\\out" + image)
                for object in objects_detected:
                    print(image + " object: " + object["name"] + " probability: " + str(object["percentage_probability"]))


def main():
    """
    Driver function, calls the class instance and drives the program
    :return: None
    """
    if(len(sys.argv)) is 4:
        drive = object_detector(sys.argv[1], sys.argv[2], sys.argv[3])
        drive.detect()
        print("\n\n  ___Done please check output_image_folder for outputs___")
    else:
        print("Please check arguments ")
        print("1)model_folder 2)test_images_folder 3)output_folder")


if __name__ == "__main__":
    main()
