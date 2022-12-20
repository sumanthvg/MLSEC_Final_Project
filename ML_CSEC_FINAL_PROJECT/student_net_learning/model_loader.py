

def get_model_net(model_name):
    from student_net_learning.models.vgg19 import vgg19
    print('Loading VGG 19')
    net = vgg19(pretrained=True)
    return net
