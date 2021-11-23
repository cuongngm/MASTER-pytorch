import torch
import numpy as np


class BaseConvertor:
    def __init__(self, dict_file=None):
        self.idx2char = []
        with open(dict_file, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip('\n')
                if line != '':
                    self.idx2char.append(line)
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx
        self.n_classes = len(self.idx2char)

    def str2idx(self, strings):
        """
        convert strings to indexes
        """
        assert isinstance(strings, list)
        indexes = []
        for string in strings:
            index = []
            for char in string:
                char_idx = self.char2idx.get(char)
                if char_idx is None:
                    raise Exception (f'Character: {char} not in dict')
                index.append(char_idx)
            indexes.append(index)
        return indexes

    def idx2str(self, indexes):
        assert isinstance(indexes, list)
        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            string.append(','.join(string))
        return strings


class TableMasterConvertor(BaseConvertor):
    def __init__(self, dict_file, max_seq_len, start_end_same, with_unk):
        super().__init__()
        self.pad_idx = 0
        self.end_idx = 1
        self.sos_idx = 2
        self.unk_idx = 3
        self.max_seq_len = max_seq_len

    def str2tensor(self, strings):
        assert isinstance(strings, list)
        tensors, padded_tgts = [], []
        indexes = self.str2idx(strings)
        for index in indexes:
            tensor = torch.LongTensor(index)
            tensors.append(tensor)
            src_seq = torch.LongTensor(tensor.size(0) + 2).fill_(0)
            src_seq[-1] = self.end_idx
            src_seq[0] = self.sos_idx
            src_seq[1:-1] = tensor
            padded_tgt = (torch.ones(self.max_seq_len) * self.pad_idx).long()
            char_num = src_seq.size(0)
            if char_num > self.max_seq_len:
                padded_tgt = src_seq[:self.max_seq_len]
            else:
                padded_tgt[:char_num] = src_seq
            padded_tgts.append(padded_tgt)
        padded_tgts = torch.stack(padded_tgts, 0).long()
        return {'targets': tensors, 'padded_targets': padded_tgts}

    def tensor2idx(self, outputs, img_metas=None):
        """
        convert ouput table master to text-index
        """
        batch_size = outputs.size(0)
        ignore_indexes = [self.pad_idx]
        indexes, scores = [], []
        for idx in range(batch_size):
            seq = outputs[idx, :, :]
            seq = seq.softmax(-1)
            max_value, max_idx = torch.max(seq, -1)
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    continue
                if char_index == self.sos_idx:
                    break
                str_index.append(char_index)
                str_score.append(char_score)
            indexes.append(str_index)
            scores.append(str_score)
        return indexes, scores

    def idx2str(self, indexes):
        assert isinstance(indexes, list)
        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            string = ','.join(string)
            strings.append(string)
        return strings

    def get_string_scores(self, str_scores):
        string_scores = []
        for str_score in str_scores:
            score = sum(str_score) / len(str_score)
            string_scores.append(score)
        return string_scores

    def get_pred_bbox_mask(self, strings):
        """
        get the bbox mask by the pred strings results, <td></td> set to 1, others set to 0
        """
        assert isinstance(strings, list)
        pred_bbox_masks = []
        SOS = self.idx2char[self.sos_idx]
        EOS = self.idx2char[self.end_idx]
        PAD = self.idx2char[self.pad_idx]
        for string in strings:
            pred_bbox_mask = []
            char_list = string.split(',')
            for char in char_list:
                if char == EOS:
                    pred_bbox_mask.append(0)
                    break
                elif char == PAD or char == SOS:
                    pred_bbox_mask.append(0)
                    continue
                else:
                    if char == '<td></td>' or char == '<td':
                        pred_bbox_mask.append(1)
                    else:
                        pred_bbox_mask.append(0)
            pred_bbox_masks.append(pred_bbox_mask)
        return np.array(pred_bbox_masks)

    def filter_invalid_bbox(self, output_bbox, pred_bbox_mask):
        low_mask = (output_bbox >= 0.) * 1.
        high_mask = (output_bbox <= 1.) * 1.
        mask = np.sum((low_mask + high_mask), axis=1)
        value_mask = np.where(mask == 2*4, 1, 0)  # 1 or 0

        output_bbox_len = output_bbox.shape[0]
        pred_bbox_mask_len = pred_bbox_mask.shape[0]
        padded_pred_bbox_mask = np.zeros(output_bbox_len, dtype='int64')
        padded_pred_bbox_mask[:pred_bbox_mask_len] = pred_bbox_mask
        filtered_output_bbox = output_bbox * np.expand_dims(value_mask, 1) * np.expand_dims(padded_pred_bbox_mask, 1)
        return filtered_output_bbox

    def decode_bboxes(self, outputs_bbox, pred_bbox_masks, img_metas):
        """
        De-normalize and scale back the box coordinate
        """
        pred_bboxes = []
        for output_bbox, pred_bbox_mask, img_meta in zip(outputs_bbox, pred_bbox_masks, img_metas):
            output_bbox = output_bbox.cpu().numpy()
            scale_factor = img_meta['scale_factor']
            pad_shape = img_meta['pad_shape']
            ori_shape = img_meta['ori_shape']

            output_bbox = self.filter_invalid_bbox(output_bbox, pred_bbox_mask)
            output_bbox[:, 0::2] = output_bbox[:, 0::2] * pad_shape[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] * pad_shape[0]

            output_bbox[:, 0::2] = output_bbox[:, 0::2] / scale_factor[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] / scale_factor[0]

            pred_bboxes.append(output_bbox)
        return pred_bboxes

    def adjust_bboxes_len(self, bboxes, strings):
        new_bboxes = []
        for bbox, string in zip(bboxes, strings):
            string = string.split(',')
            string_len = len(string)
            bbox = bbox[:string_len, :]
            new_bboxes.append(bbox)
        return new_bboxes

    def str_bbox_format(self, img_metas):
        """
        Convert text-string into tensor.
        Pad 'bbox' and 'bbox_masks' to the same length as 'text'

        Args:
            img_metas (list[dict]):
                dict.keys() ['filename', 'ori_shape', 'img_shape', 'text', 'scale_factor', 'bbox', 'bbox_masks']
        Returns:
            dict (str: Tensor | list[tensor]):
                tensors (list[Tensor]): [torch.Tensor([1,2,3,3,4]),
                                                    torch.Tensor([5,4,6,3,7])]
                padded_targets (Tensor(bsz * max_seq_len))

                bbox (list[Tensor]):
                bbox_masks (Tensor):
        """

        # output of original str2tensor function(split by ',' in each string).
        gt_labels = [[char for char in img_meta['text'].split(',')] for img_meta in img_metas]
        tmp_dict = self.str2tensor(gt_labels)
        text_target = tmp_dict['targets']
        text_padded_target = tmp_dict['padded_targets']

        # pad bbox's length
        bboxes = [img_meta['bbox'] for img_meta in img_metas]
        bboxes = self._pad_bbox(bboxes)

        # pad bbox_mask's length
        bbox_masks = [img_meta['bbox_masks'] for img_meta in img_metas]
        bbox_masks = self._pad_bbox_mask(bbox_masks)

        format_dict = {'targets': text_target,
                       'padded_targets': text_padded_target,
                       'bbox': bboxes,
                       'bbox_masks': bbox_masks}

        return format_dict

    def output_format(self, output, bbox_output, img_metas=None):
        # cls_branch
        str_indexes, str_scores = self.tensor2idx(output, img_metas)
        strings = self.idx2str(str_indexes)
        scores = self.get_string_scores(str_scores)
        # bbox_branch
        pred_bbox_masks = self.get_pred_bbox_mask(strings)
        pred_bboxes = self.decode_bboxes(bbox_output, pred_bbox_masks, img_metas)
        pred_bboxes = self.adjust_bboxes_len(pred_bboxes, strings)
        return strings, pred_bboxes
