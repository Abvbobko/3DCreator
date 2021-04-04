import PIL.Image
import PIL.ExifTags

data_path = "C:\\Users\\hp\\Desktop\\3DCreator\\creator_3d\\data\\rock_head\\"
image_1_path = data_path + "1.jpg"

img = PIL.Image.open(image_1_path)
exif_data = img._getexif()
exif = {
    PIL.ExifTags.TAGS[k]: v
    for k, v in img._getexif().items()
    if k in PIL.ExifTags.TAGS
}
for k in exif.keys():
    print(f"{k}: {exif[k]}")

print("!!!", img.getexif())
