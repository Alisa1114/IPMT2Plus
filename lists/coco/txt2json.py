import json

train_list = []
with open("train_data_list.txt", "r") as f:
    for line in f.readlines():
        image_path = line.split(" ")[0]
        label_path = line.split(" ")[1]
        image_name = image_path.split("/")[-1]
        label_name = label_path.split("/")[-1]
        image_name = "train2014/" + image_name
        label_name = "annotations/coco_masks/instance_train2014/" + label_name
        image_name = image_name.strip()
        label_name = label_name.strip()
        print(image_name, label_name)
        train_list.append([image_name, label_name])

with open("train_list.json", "w") as f:
    json.dump(train_list, f)

val_list = []
with open("val_data_list.txt", "r") as f:
    for line in f.readlines():
        image_path = line.split(" ")[0]
        label_path = line.split(" ")[1]
        image_name = image_path.split("/")[-1]
        label_name = label_path.split("/")[-1]
        image_name = "val2014/" + image_name
        label_name = "annotations/coco_masks/instance_val2014/" + label_name
        image_name = image_name.strip()
        label_name = label_name.strip()
        print(image_name, label_name)
        val_list.append([image_name, label_name])

with open("val_list.json", "w") as f:
    json.dump(val_list, f)

print(
    "number of train data:{}\nnumber of val data:{}".format(
        len(train_list), len(val_list)
    )
)
