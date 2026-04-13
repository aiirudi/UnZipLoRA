from typing import Optional, Union, List
import numpy as np
import torch
from torch import nn
import copy

class UnZipLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int,
        rank: int = 64,
        lora_matrix_key: List[str] = None, 
        device: Optional[Union[torch.device, str]] = None,
        # dtype: Optional[torch.dtype] = torch.float32,
        dtype: Optional[torch.dtype] = None,
        **model_kwargs
    ):
        super().__init__()


        self.lora_matrix_dic = nn.ModuleDict() # 用于存储LoRA 的矩阵。每个key是一个矩阵的名称， value 则是存储的矩阵
        self.fixed_matrix = {} 
        self.lora_matrix_dic_norm = {} # 存储每个 LoRA 矩阵的范数
        # * If masked matrix => True: the column filter is used / not all columns are used
        
        # 用于实现 block separation
        # *                 => False: the filter is not used => all coluns are used
        self.masked_matrix = {} # 判断是否有 mask，用于判断哪些列是有用的1
        
        for key in lora_matrix_key: # 用于生成 content_LoRA 和 style_LoRA 的 down 和 up 矩阵
            self.lora_matrix_dic[f"{key}_down"] = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
            self.lora_matrix_dic[f"{key}_up"] = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
            # setattr(self, f"merge_{key}", nn.Parameter(torch.ones((out_features,), device=device, dtype=dtype), requires_grad=True))
            nn.init.normal_(self.lora_matrix_dic[f"{key}_down"].weight, std=1 / rank)
            nn.init.normal_(self.lora_matrix_dic[f"{key}_up"].weight, std=1 / rank)
            self.lora_matrix_dic_norm[f"{key}_norm_down"] = torch.norm(self.lora_matrix_dic[f"{key}_down"].weight.detach(), dim=0, keepdim=True)
            self.lora_matrix_dic_norm[f"{key}_norm_up"] = torch.norm(self.lora_matrix_dic[f"{key}_up"].weight.detach(), dim=0, keepdim=True)
            # * Whether use column filter, initialized: do not use
            self.masked_matrix[key] = False
        
        # TODO: hard code for only one content and one style
        # * merge: the value for each columns
        # * mask: the column filter(bool)
        
        # merge_{key} 是软 mask (已修改)
        self.merge_content = nn.Parameter(torch.ones(rank, device=device, dtype=dtype, requires_grad=True))
        self.merge_style = nn.Parameter(torch.ones(rank, device=device, dtype=dtype, requires_grad=True))
        
        # 计算每一列的重要性(已修改)
        self.column_score_content = torch.ones(rank)
        self.column_score_style = torch.ones(rank)
        
        # 用于实现 mask separation (已修改)
        self.mask_content = torch.zeros(rank, device=device, dtype=torch.bool)
        self.mask_style = torch.zeros(rank, device=device, dtype=torch.bool)
        
        self.lora_matrix_key = lora_matrix_key
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.dtype = dtype
        self.forward_type = "both"
        self.device = device
    
    def set_cone_score(self, key):
        setattr(self, f"column_score_{key}", torch.zeros(self.rank))
    
    def set_forward(self, type: str = "both"):
        assert type in ["both", "content", "style"]
        self.forward_type = type
        
    def compute_mergers_similarity(self):
        # * If not filtered, directly compute the cosine similarity
        if self.masked_matrix["content"] is False or self.masked_matrix["style"] is False:
            return (self.merge_content * self.merge_style).abs().mean().unsqueeze(0)
        # * If filtered, first combined with mask and then compute the coisne similarity
        else:    
            return ((self.merge_content * self.mask_content) * (self.merge_style * self.mask_style)).abs().mean().unsqueeze(0)
    
    def set_merger_gradient(self, key, value=False):
        merger_matrix = getattr(self, f"merge_{key}")
        merger_matrix.requires_grad = value
        setattr(self, f"merge_{key}", merger_matrix)
    
    # * Each merger should between (0, 1)
    def clamp_merger(self, key):
        merge_matrix = getattr(self, f"merge_{key}")
        merge_matrix.clamp_(0, 1)
        setattr(self, f"merge_{key}", merge_matrix)
    
    def get_merger_mask(self, key):
        merge_matrix = getattr(self, f"merge_{key}")
        return merge_matrix
            
    def set_layer_mask(self, key, value=True):
        self.masked_matrix[key] = value

    def get_unziplora_norm(self, key, dim="L2", quick_log=False, multiple=True):
        if multiple is True: 
            merge_matrix = getattr(self, f"merge_{key}")
            # merge_matrix = (torch.tanh(merge_matrix) + 1) / 2 #* merge_matrix
            
            D = self.lora_matrix_dic[f"{key}_down"].weight.T * merge_matrix
            U = self.lora_matrix_dic[f"{key}_up"].weight.data.T
        else: 
            D = self.lora_matrix_dic[f"{key}_down"].weight.T
            U = self.lora_matrix_dic[f"{key}_up"].weight.data.T 
                    
        if self.masked_matrix[key] is True:
            D *= getattr(self, f"mask_{key}")
        
        merged_matrix = D @ U

        if dim == "L2":
            norm = torch.norm(merged_matrix, p="fro")
        elif dim == "L1":
            norm = torch.norm(merged_matrix, p=1)
        elif dim == "nuc":
            norm = torch.norm(merged_matrix, p='nuc')
        norm_return = torch.tensor(norm.item()).to(self.device) if quick_log else norm.unsqueeze(0)
        return norm_return 
    

    # 现在知道这个是返回加权权重就好了
    def get_unziplora_weight(self, key):
        # print(self.lora_matrix_dic.keys())
        # * Get weight without filter
        merge_matrix = getattr(self, f"merge_{key}")
        if self.masked_matrix[key] is False:
            # merge_matrix = (torch.tanh(merge_matrix) + 1) / 2#* merge_matrix  
            # return self.lora_matrix_dic[f"{key}_down"].weight.data, self.lora_matrix_dic[f"{key}_up"].weight.data
            return self.lora_matrix_dic[f"{key}_down"].weight.data * merge_matrix.unsqueeze(1), self.lora_matrix_dic[f"{key}_up"].weight.data
        else:
            # 这里 unsqueeze(1) 是因为这里是逐元素相乘
            filter_matrix = getattr(self, f"mask_{key}")
            return self.lora_matrix_dic[f"{key}_down"].weight.data * filter_matrix.unsqueeze(1), self.lora_matrix_dic[f"{key}_up"].weight.data
                 
    
    def get_unziplora_cone(self, key, accumulate=True):
        '''
        Compute cone value for both style and content, store the value in self.column_score
        Will be used when all columns are used ==> The computed cone will help determine which columns \
        will be used in following training ==> the filter will not included in computation
        Theratically(if no bugs), every parameters except merger will have gradient
        '''
        merge_matrix = getattr(self, f"merge_{key}") #merge_matrix.shape: (out_features)
        merger_gradient = merge_matrix.grad # (out_features, )
        
        """
        # 考虑合并后的梯度， 原代码
        if merger_gradient is None: 
            if self.lora_matrix_dic[f"{key}_down"].weight.grad is None:
                merged_gradient = self.lora_matrix_dic[f"{key}_down"].weight.data.T @ self.lora_matrix_dic[f"{key}_up"].weight.grad.T * merge_matrix        
            else:
                merged_gradient = self.lora_matrix_dic[f"{key}_down"].weight.grad.T @ self.lora_matrix_dic[f"{key}_up"].weight.data.T * merge_matrix +\
                                self.lora_matrix_dic[f"{key}_down"].weight.data.T @ self.lora_matrix_dic[f"{key}_up"].weight.grad.T * merge_matrix
        else:
            if self.lora_matrix_dic[f"{key}_down"].weight.grad is None:
                merged_gradient = self.lora_matrix_dic[f"{key}_down"].weight.data.T @ self.lora_matrix_dic[f"{key}_up"].weight.grad.T * merge_matrix + \
                                merged_weight * merger_gradient
            else:
                merged_gradient = self.lora_matrix_dic[f"{key}_down"].weight.grad.T @ self.lora_matrix_dic[f"{key}_up"].weight.data.T * merge_matrix +\
                                self.lora_matrix_dic[f"{key}_down"].weight.data.T @ self.lora_matrix_dic[f"{key}_up"].weight.grad.T * merge_matrix + \
                                merged_weight * merger_gradient
        """
        if merger_gradient is None:
            if self.lora_matrix_dic[f"{key}_down"].weight.grad is None:
                dL_dU = self.lora_matrix_dic[f"{key}_down"].weight.data.T  * merge_matrix.view(1, -1)
                U = self.lora_matrix_dic[f"{key}_up"].weight.grad.T

                merged_gradient = dL_dU.sum(dim=0) * U.sum(dim=1)
            else:
                D = self.lora_matrix_dic[f"{key}_down"].weight.data.T * merge_matrix.view(1, -1)
                U = self.lora_matrix_dic[f"{key}_up"].weight.data.T

                dL_dD = self.lora_matrix_dic[f"{key}_down"].weight.grad.T * merge_matrix.view(1, -1)
                dL_dU = self.lora_matrix_dic[f"{key}_up"].weight.grad.T
                
                merged_gradient = dL_dD.sum(dim=0) * U.sum(dim=1) + D.sum(dim=0) * dL_dU.sum(dim=1)
        else:
            if self.lora_matrix_dic[f"{key}_down"].weight.grad is None:
                D = self.lora_matrix_dic[f"{key}_down"].weight.data.T * merge_matrix.view(1, -1)
                U = self.lora_matrix_dic[f"{key}_up"].weight.data.T
                
                dL_dD = self.lora_matrix_dic[f"{key}_down"].weight.data.T * merger_gradient.view(1, -1)
                dL_dU = self.lora_matrix_dic[f"{key}_up"].weight.grad.T
            
                merged_gradient = dL_dD.sum(dim=0) * U.sum(dim=1) + D.sum(dim=0) * dL_dU.sum(dim=1)
            else:
                D = self.lora_matrix_dic[f"{key}_down"].weight.data.T * merge_matrix.view(1, -1)
                U = self.lora_matrix_dic[f"{key}_up"].weight.data.T
                
                dD_dB = self.lora_matrix_dic[f"{key}_down"].weight.grad.T * merge_matrix.view(1, -1)
                dD_dmerge = self.lora_matrix_dic[f"{key}_down"].weight.data.T * merger_gradient.view(1, -1)
                dL_dD = dD_dB.sum(dim=0) + dD_dmerge.sum(dim=0)
                dL_dU = self.lora_matrix_dic[f"{key}_up"].weight.grad.T
            
                merged_gradient = dL_dD * U.sum(dim=1) + D.sum(dim=0) * dL_dU.sum(dim=1)
    
        D = self.lora_matrix_dic[f"{key}_down"].weight.data.T
        U = self.lora_matrix_dic[f"{key}_up"].weight.data.T

        # cone.shape (rank,)
        cone = D.sum(dim=0) * merged_gradient * U.sum(dim=1)

        # 如果是累积，则就是对每个 lora layer 统计 cone
        if accumulate: 
            # 下面这个是 abs-cone 方法， 先尝试原方法在尝试 abs-cone 方法
            # setattr(self, f"column_score_{key}", getattr(self, f"column_score_{key}").to(self.device) + torch.abs(cone.to(self.device)))
            setattr(self, f"column_score_{key}", getattr(self, f"column_score_{key}").to(self.device) + cone.to(self.device))
        else: 
            # 如果不是累积那么就是要选列了，torch.abs(cone) > 1e-5 是用来判断列的“活跃度”
            # 每一列的非 0 数的总和/行数， 就代表这个列的活跃度
            cone_sparsity = (torch.abs(cone) > 1e-5).float() / cone.shape[0]
            setattr(self, f"column_score_{key}", cone_sparsity)
    
    """
    对 merge_content / merge_style 的梯度"按列做门控", 控制哪些列的 merge_content 和 merge_style 可以更新。
    """
    def set_gradient_mask(self, finetune_mask):
        '''
            set the gradient map for column features
            For up layers: if the filter is selected(if not been masked is True), trained
            if set finetune_mask as false: only overlapped part is trained
                                 as true : all are trained
            For merger: if the two are overlapped with each other, the merger will be trained
            Only called if only the overlapped part are trained
        '''
        merge_content = getattr(self, f"merge_content")
        merge_style = getattr(self, f"merge_style")

        merger_overlapped = self.mask_content & self.mask_style
        # print(merger_overlapped)
        
        # merge 向量是在两个 LoRA 共用列时做权重分配， 所以 finetune_mask=False
        # 只在真正冲突（重叠）的列上优化merge, 
        # fintune_mask=True,更激进，进——让每个LoRA在自己所有选中列上都优化merge缩放
        if merge_content.grad is not None and merge_style.grad is not None:
            # finetune_mask 默认是 False
            if not finetune_mask:
                merge_content.grad *= merger_overlapped 
                merge_style.grad *= merger_overlapped 
            else: 
                merge_content.grad *= self.mask_content
                merge_style.grad *= self.mask_style 
                
            setattr(self, "merge_content", merge_content)
            setattr(self, "merge_style", merge_style)
    
    # * use hard mask and creat filters from top k columns
    def mask_updated_elements(self, key=None, step_ratio=0.1, avoid=True):
        '''
        key:None/"content"/"style" 决定给谁做列许纳泽
        step_ratio:每轮选择比例,默认 0.1
        avoid:style 选列时是否避开content 已选的列
        '''
        '''
            # Now we will set the mask == previous + current
            
            Args:
            key: compute the sparse masks for given keys 
                key = None: sparse masks for both content and style 
                key = content / style: the given keys will have sparse mask while features of 
                                       the other will train all 
            step_ratio: how many columns are seleted
            avoid: True: will choose both content and style but content is prioritized 
        
        '''
        # 首先计算选择的列数
        selected_num = int(self.rank * step_ratio) # 已修改
        
        if key is None: 
            # 这是大多数层的情况--该层不再 block separation 的任何一方独占列表中
            # if key is none, no blocks are masked, i.e: will generate columns mask for both content and style
            # style mask is the Complement of content
            
            # 从列稀疏度分数中抽取 selected_num 中选 top-k，
            top_content_values, _= torch.topk(self.column_score_content, selected_num)
            # _, top_indices_style = torch.topk(getattr(self, f"column_score_style"), top_k)

            # 取选中的值中的最小值作为阈值
            if top_content_values.numel() > 0:
                threshold = top_content_values.min()
            else:
                threshold = float('inf')
            
            # 大于阈值的列标为 True， 小于阈值的列标为 False
            content_mask_current = self.column_score_content > threshold 
            # 与历史 mask 取或保证不会丢失
            self.mask_content = content_mask_current | self.mask_content
            
            #  克隆历史 style 的mask, 之前是对 content mask 操作
            masked_style = self.column_score_style.clone()
            
            # 在 avoid 为 True 的时候， 就不可能选到之前 content 选中的列
            if avoid:
                masked_style[self.mask_content] = float('-inf') # 把之前 content 选中的列发分数设置为 -inf， 保证一定无法选中，这就保证 content LoRA 和 style LoRA 的列一定不会重合。
            
            # 之类的操作和 content 一样选中前 selected_num 列， 
            top_style_values, _= torch.topk(masked_style, selected_num)
            if top_style_values.numel() > 0:
                threshold = top_style_values.min()
            else:
                threshold = float('inf')
            # 将选中的列设置为 True
            mask_style_current = masked_style > threshold 
            # 将当前选中的和历史选中的 style mask 取或，保证之前的不丢失。
            self.mask_style = mask_style_current | self.mask_style
        else:
            # * generate sparse mask for given key 
            # 这里也是同理，不过只会对当前传入的key 做 mask选择, key 是被稀疏的列
            top_values, _= torch.topk(getattr(self, f"column_score_{key}"), selected_num)
            if top_values.numel() > 0:
                threshold = top_values.min()
            else:
                threshold = float('inf')
            mask = getattr(self, f"column_score_{key}") > threshold 
            setattr(self, f"mask_{key}", mask | getattr(self, f"mask_{key}"))
            
            # 另一方就全开，表示可以直接训练
            all_on_key = "content" if key == "style" else "style"
            setattr(self, f"mask_{all_on_key}", torch.ones(self.rank, device=self.device, dtype=torch.bool))
    
    def log_selected_mask(self, key):
        return getattr(self, f"column_score_{key}") * getattr(self, f"mask_{key}")
    
    def forward(self, hidden_states_content: torch.Tensor, hidden_states_style: torch.Tensor=None) -> torch.Tensor:
        '''
        forward with content and style hidden states 
        the weight depend on soft mask self.merge and hard mask(bor block separation) self.mask
        if set forward type as both: content and style are used
        '''
        dtype = self.dtype
    
        # 对于
        if self.forward_type == "both":
            orig_dtype = hidden_states_content.dtype
                # print(hidden_states.shape)
            # 没传入 style 的 hidden_states 就用 content 的 hidden_states
            if hidden_states_style is None: 
                hidden_states_style = hidden_states_content
            
            # 对 content_lora_weight 使用 软mask
            D_content = self.lora_matrix_dic["content_down"].weight.T * self.merge_content
            U_content = self.lora_matrix_dic["content_up"].weight.T 
            
            
            # 如果硬 mask 开启，则再乘上 硬mask
            if self.masked_matrix["content"] is True:
                D_content *= self.mask_content
            masked_content_weight = D_content @ U_content

            # 输入乘以最终权重 content_text_prompt_hidden_states (B, seq, in) @ masked_content_weight : (in, out) -> up_hidden_states_content (B, seq, outs)
            
            up_hidden_states_content = hidden_states_content.to(dtype) @ masked_content_weight
            
            # 和上段同理，这里就是变成了对 style_lora_weight操作而已
            D_style = self.lora_matrix_dic["style_down"].weight.T * self.merge_style
            U_style = self.lora_matrix_dic["style_up"].weight.T 
            
            if self.masked_matrix["style"] is True: 
                D_style *= self.mask_style
            masked_style_weight = D_style @ U_style
        
            # 这里也是输入乘以最终权重，只不过变成了是在计算 style 的而已
            up_hidden_states_style = hidden_states_style.to(dtype) @ masked_style_weight
            added_hidden_states = up_hidden_states_style.to(orig_dtype) + up_hidden_states_content.to(orig_dtype)
        
        # 如果 forward_type 是 "content"，那么就只计算经过 content 的 值 
        if self.forward_type == "content":
            orig_dtype = hidden_states_content.dtype

            D_content = self.lora_matrix_dic["content_down"].weight.T * self.merge_content
            U_content = self.lora_matrix_dic["content_up"].weight.T

            if self.masked_matrix["content"] is True:
                D_content = D_content * self.mask_content
            merged_content_weight = D_content @ U_content
            up_hidden_states_content = hidden_states_content.to(dtype) @ merged_content_weight
            added_hidden_states = up_hidden_states_content.to(orig_dtype)
        
        # 这个同理，就只单独计算 style 的 text_hidden_states 的值
        if self.forward_type == "style":
            orig_dtype = hidden_states_style.dtype

            D_style = self.lora_matrix_dic["style_down"].weight.T * self.merge_style
            U_style = self.lora_matrix_dic["style_up"].weight.T


            if self.masked_matrix["style"] is True:
                D_style = D_style * self.mask_style
            merged_style_weight = D_style @ U_style
            up_hidden_states_style = hidden_states_style.to(dtype) @ merged_style_weight
            added_hidden_states = up_hidden_states_style.to(orig_dtype)
        return added_hidden_states
    
class UnZipLoRALinearLayerInfer(nn.Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        rank: int = 64,
        lora_matrix_key: List[str] = None, 
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.lora_matrix_dic = nn.ModuleDict()
        
        # y用于控制 block separation
        self.masked_matrix = {}
        
        
        for key in lora_matrix_key:
            self.lora_matrix_dic[f"{key}_down"] = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
            self.lora_matrix_dic[f"{key}_up"] = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
            nn.init.normal_(self.lora_matrix_dic[f"{key}_down"].weight, std=1 / rank)
            nn.init.normal_(self.lora_matrix_dic[f"{key}_up"].weight, std=1 / rank)
            self.masked_matrix[key] = False
        self.lora_matrix_key = lora_matrix_key
        self.out_features = out_features
        self.forward_type = "both"
        self.dtype = dtype

        # 软 mask
        self.merge_content = nn.Parameter(torch.ones(rank, device=device, dtype=dtype))
        self.merge_style = nn.Parameter(torch.ones(rank, device=device, dtype=dtype))

    def set_layer_mask(self, key, value=True):
        self.masked_matrix[key] = value
    
    def set_forward(self, type: str = "both"):
        assert type in ["both", "content", "style"]
        self.forward_type = type

    def forward(self, hidden_states_content: torch.Tensor, hidden_states_style: torch.Tensor=None) -> torch.Tensor:
        dtype = self.dtype
        merged_content = self.merge_content
        merged_style = self.merge_style
        
        if self.forward_type == "both":
            orig_dtype = hidden_states_content.dtype
                # print(hidden_states.shape)
            if hidden_states_style is None: 
                hidden_states_style = hidden_states_content
            if self.masked_matrix["content"] is True:
                up_hidden_states_content = torch.zeros((hidden_states_content.shape[0], hidden_states_content.shape[1], self.out_features)).to(hidden_states_content.device)
            else: 
                D_content = self.lora_matrix_dic["content_down"].weight.T * merged_content
                U_content = self.lora_matrix_dic["content_up"].weight.T 

                masked_content_weight = D_content @ U_content
                up_hidden_states_content = hidden_states_content.to(dtype) @ masked_content_weight
            
            if self.masked_matrix["style"] is True: 
                up_hidden_states_style = torch.zeros((hidden_states_style.shape[0], hidden_states_style.shape[1], self.out_features)).to(hidden_states_style.device)
            else: 
                D_style = self.lora_matrix_dic["style_down"].weight.T * merged_style
                U_style = self.lora_matrix_dic["style_up"].weight.T 
                
                masked_style_weight = D_style @ U_style

                up_hidden_states_style = hidden_states_style.to(dtype) @ masked_style_weight
            added_hidden_states = up_hidden_states_style.to(orig_dtype) + up_hidden_states_content.to(orig_dtype)
        
        if self.forward_type == "content":
            orig_dtype = hidden_states_content.dtype
            if self.masked_matrix["content"] is True:
                up_hidden_states_content = torch.zeros((hidden_states_content.shape[0], hidden_states_content.shape[1], self.out_features)).to(hidden_states_content.device)
            else: 
                D_content = self.lora_matrix_dic["content_down"].weight.T * merged_content
                U_content = self.lora_matrix_dic["content_up"].weight.T 
                                        
                masked_content_weight = D_content @ U_content

                up_hidden_states_content = hidden_states_content.to(dtype) @ masked_content_weight
            added_hidden_states = up_hidden_states_content.to(orig_dtype)
        if self.forward_type == "style":
            orig_dtype = hidden_states_style.dtype
            if self.masked_matrix["style"] is True: 
                up_hidden_states_style = torch.zeros((hidden_states_style.shape[0], hidden_states_style.shape[1], self.out_features)).to(hidden_states_style.device)
            else: 
                D_style = self.lora_matrix_dic["style_down"].weight.T * merged_style
                U_style = self.lora_matrix_dic["style_up"].weight.T 

                masked_style_weight =  D_style @ U_style

                up_hidden_states_style = hidden_states_style.to(dtype) @ masked_style_weight
            added_hidden_states = up_hidden_states_style.to(orig_dtype)
        return added_hidden_states
