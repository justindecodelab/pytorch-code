from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

img = Image.open('./cat.jpg')

# 数据增强示例
# transform = transforms.Compose([transforms.CenterCrop(10), transforms.ToTensor()])

center_crop = transforms.CenterCrop([200, 200])(img)
random_crop = transforms.RandomCrop([200, 200])(img)
random_resized_crop = transforms.RandomResizedCrop(200,
                                                   scale=(0.08, 1.0),
                                                   ratio=(0.75, 1.55),
                                                   interpolation=2)(img)

h_flip = transforms.RandomHorizontalFlip(0.7)(img)
v_flip = transforms.RandomVerticalFlip(0.8)(img)
random_rotation = transforms.RandomRotation(30)(img)

pad = transforms.Pad(10, fill=0, padding_mode='constant')(img)
color_jitter = transforms.ColorJitter(brightness=1,
                                      contrast=0.5,
                                      saturation=0.5,
                                      hue=0.4)(img)
gray = transforms.Grayscale(1)(img)
random_affine = transforms.RandomAffine(
    degrees=45,
    translate=(0.5, 0.7),
    scale=(0.5, 0.8),
    shear=3,
)(img)
resize = transforms.Resize([100,200])(img)
mean = [0.45,0.5,0.5]
std = [0.3,0.6,0.5]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean,std),
                                transforms.ToPILImage() # 可视化需要转换回PIL Image
                                ])

img_transform = transform(img)



def show_image_with_axes(position, title, image):
    width, height = image.size
    plt.subplot(2, 3, position)
    plt.title(f'{title} ({width}x{height})')
    plt.imshow(image)
    plt.xlabel(f'X (pixels: 0-{width - 1})')
    plt.ylabel(f'Y (pixels: 0-{height - 1})')
    plt.xticks(list(range(0, width, 50)))
    plt.yticks(list(range(0, height, 50)))


plt.figure('transform demo')

# show_image_with_axes(1, 'original', img)
# show_image_with_axes(1, 'center crop', center_crop)
# show_image_with_axes(2, 'random crop', random_crop)
# show_image_with_axes(3, 'random resized crop', random_resized_crop)
# show_image_with_axes(4, 'horizontal flip', h_flip)
# show_image_with_axes(5, 'vertical flip', v_flip)    
# show_image_with_axes(6, 'random rotation', random_rotation)
show_image_with_axes(1, 'pad', pad)
show_image_with_axes(2, 'color jitter', color_jitter)
show_image_with_axes(3, 'grayscale', gray)
show_image_with_axes(4, 'random affine', random_affine)
show_image_with_axes(5, 'resize', resize)
show_image_with_axes(6, 'normalize', img_transform)
plt.tight_layout()
plt.show()