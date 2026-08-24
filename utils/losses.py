
import torch
import torch.nn.functional as F
def loss_fucntion(a, b, L2): 
    cos_loss = torch.nn.CosineSimilarity()
    l2_loss = torch.nn.MSELoss()
    loss = 0
    
    # Use cosloss only
    if L2 == 0:
        for item in range(len(a)): 
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)))  
    
    # Use l2loss and cosloss
    if L2 == 2:
        for item in range(len(a)):
             loss += 0.5*torch.mean(l2_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))       
             loss += 0.5*torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),b[item].view(b[item].shape[0],-1)))

    # Use l2loss only
    if L2 == 1:
        l2_loss = torch.nn.MSELoss()
        for item in range(len(a)):
             loss += torch.mean(l2_loss(a[item].reshape(a[item].shape[0],-1),b[item].reshape(b[item].shape[0],-1)))       

    loss2 = loss_fucntion_2(a, b)

    return loss, loss2

def loss_fucntion_2(a, b): 
    mse_loss = torch.nn.MSELoss()

    a2 = F.interpolate(a[2], size=64, mode='bilinear', align_corners=True)
    b2 = F.interpolate(b[2], size=64, mode='bilinear', align_corners=True)
    l1_1 = torch.mean(mse_loss(a2.reshape(a2.shape[0],-1),b2.reshape(b2.shape[0],-1)))
    l1_2 = torch.mean(mse_loss(a[1].reshape(a[1].shape[0],-1),b[1].reshape(b[1].shape[0],-1)))
    loss2_1 = torch.abs(l1_1 - l1_2)
    
    a1 = F.interpolate(a[1], size=64, mode='bilinear', align_corners=True)
    b1 = F.interpolate(b[1], size=64, mode='bilinear', align_corners=True)
    l2_1 = torch.mean(mse_loss(a1.reshape(a1.shape[0],-1),b1.reshape(b1.shape[0],-1)))
    l2_2 = torch.mean(mse_loss(a[0].reshape(a[0].shape[0],-1),b[0].reshape(b[0].shape[0],-1)))
    loss2_2 = torch.abs(l2_1 - l2_2)
    
    double_a2 = F.interpolate(a2, size = 64, mode='bilinear', align_corners=True)
    double_b2 = F.interpolate(b2, size = 64, mode='bilinear', align_corners=True)
    l3_1 = l2_1 = torch.mean(mse_loss(double_a2.reshape(double_a2.shape[0],-1), double_b2.reshape(double_b2.shape[0],-1)))
    l3_2 = l2_2
    loss2_3 = torch.abs(l3_1 - l3_2)

    loss2 = loss2_1 + loss2_2 + loss2_3
    return loss2

def MSE_loss(a,b):

    loss = 0.
    l2_loss = torch.nn.MSELoss()
    for item in range(len(a)):
        loss += torch.mean(l2_loss(a[item].reshape(a[item].shape[0],-1),
                                   b[item].reshape(b[item].shape[0],-1)))       

    return loss


def distill_loss(a,b,T=3):

    dis_loss = 0.
    for item in range(len(a)):
        teach = F.softmax((a[item].reshape(a[item].shape[0],-1))/T, dim=1)
        stu = F.log_softmax((b[item].reshape(b[item].shape[0],-1))/T, dim=1)
        dis_loss += F.kl_div(stu, teach, reduction='batchmean') * (T ** 2)

    return dis_loss

def BCEloss(a,b):
    loss=0.
    criterion = torch.nn.BCEWithLogitsLoss()
    for item in range(len(a)):
        loss += torch.mean(criterion((a[item].reshape(a[item].shape[0],-1)), 
                          (b[item].reshape(b[item].shape[0],-1))))
    
    return loss


def KLloss(a,b):
    loss = 0.
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    for item in range(len(a)):
        loss += torch.mean(criterion(F.log_softmax(a[item].reshape(a[item].shape[0],-1), dim=1), 
                          F.softmax(b[item].reshape(b[item].shape[0],-1), dim=1)))
    
    return loss

def CosineLoss(a, b):
    loss = 0.
    cos_loss = torch.nn.CosineSimilarity()
    for item in range(len(a)):
        loss += torch.mean(cos_loss(a[item].reshape(a[item].shape[0],-1), 
                                         b[item].reshape(b[item].shape[0],-1)))
    return loss

        # loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
        #                               b[item].view(b[item].shape[0],-1)))
def loss_fucntion(a, b):

    loss = 0.
    cos_loss = torch.nn.CosineSimilarity()
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].reshape(a[item].shape[0],-1), 
                                         b[item].reshape(b[item].shape[0],-1)))
    return loss


def CEloss(a,b):
    loss = 0.
    celoss = torch.nn.CrossEntropyLoss()
    for item in range(len(a)):
        loss += torch.mean(celoss())




def DiversityLoss(features_student1, features_student2, 
                  low_dim_layers=1, tau_low=0.5, tau_high=0.8):
    """
    学生多样性优化损失 - 适配3个block的网络
    
    Args:
        features_student1: 学生网络1的多层特征列表 [block1, block2, block3]
        features_student2: 学生网络2的多层特征列表 [block1, block2, block3]
        low_dim_layers: 低维层数量（前几层）
        tau_low: 低维层相似度阈值
        tau_high: 高维层相似度阈值
    """
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    loss_div = 0.
    total_layers = len(features_student1)  # 应该是3
    
    # 第一项：低维层多样性约束（鼓励差异）
    for i in range(low_dim_layers):
        feat1 = features_student1[i].reshape(features_student1[i].shape[0], -1)
        feat2 = features_student2[i].reshape(features_student2[i].shape[0], -1)
        
        similarity = cos_sim(feat1, feat2)  # shape: [batch_size]
        # max(0, similarity - tau_low)：只在相似度超过阈值时惩罚
        loss_div += torch.mean(torch.clamp(similarity - tau_low, min=0))
    
    # 第二项：高维层一致性约束（鼓励相似）
    for i in range(low_dim_layers, total_layers):
        feat1 = features_student1[i].reshape(features_student1[i].shape[0], -1)
        feat2 = features_student2[i].reshape(features_student2[i].shape[0], -1)
        
        similarity = cos_sim(feat1, feat2)
        # -min(0, similarity - tau_high)：只在相似度低于阈值时惩罚
        loss_div -= torch.mean(torch.clamp(similarity - tau_high, max=0))
    
    return loss_div