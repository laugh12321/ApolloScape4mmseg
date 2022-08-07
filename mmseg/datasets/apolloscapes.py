'''
@File    :   apolloscapes.py
@Version :   1.0
@Author  :   laugh12321
@Contact :   laugh12321@vip.qq.com
@Date    :   2022/07/22 15:51:53
@Desc    :   None
'''
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ApolloScapesDataset(CustomDataset):
    """ApolloScapes dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_bin.png' for ApolloScapes dataset.
    """

    CLASSES = (
        'void', 's_w_d', 's_y_d', 'ds_w_dn', 'ds_y_dn', 'sb_w_do', 'sb_y_do', 'b_w_g', 'b_y_g', 'db_w_g', 'db_y_g', 'db_w_s',
        's_w_s', 'ds_w_s', 's_w_c', 's_y_c', 's_w_p', 's_n_p', 'c_wy_z', 'a_w_u', 'a_w_t', 'a_w_tl', 'a_w_tr',
        'a_w_tlr', 'a_w_l', 'a_w_r', 'a_w_lr', 'a_n_lu', 'a_w_tu', 'a_w_m', 'a_y_t', 'b_n_sr', 'd_wy_za', 'r_wy_np',
        'vom_wy_n', 'om_n_n'
    )

    PALETTE = [
        [0, 0, 0], [70, 130, 180], [220, 20, 60], [128, 0, 128], [255, 0, 0], [0, 0, 60], [0, 60, 100], [0, 0, 142], [119, 11, 32],
        [244, 35, 232], [0, 0, 160], [153, 153, 153], [220, 220, 0], [250, 170, 30], [102, 102, 156], [128, 0, 0],
        [128, 64, 128], [238, 232, 170], [190, 153, 153], [0, 0, 230], [128, 128, 0], [128, 78, 160], [150, 100, 100],
        [255, 165, 0], [180, 165, 180], [107, 142, 35], [201, 255, 229], [0, 191, 255], [51, 255, 51], [250, 128, 114],
        [127, 255, 0], [255, 128, 0], [0, 255, 255], [178, 132, 190], [128, 128, 64], [102, 0, 204]
    ]

    def __init__(self, img_suffix='.jpg', seg_map_suffix='_bin.png', **kwargs):
        super(ApolloScapesDataset, self).__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for ApolloScapes."""
        if isinstance(result, str):
            result = np.load(result)
        from dataset_api.lane_segmentation.helpers import laneMarkDetection as ASLabels
        result_copy = result.copy()
        for trainId, label in ASLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')   
            from dataset_api.lane_segmentation.helpers import laneMarkDetection as ASLabels
            palette = np.zeros((len(ASLabels.id2label), 3), dtype=np.uint8)        
            for label_id, label in ASLabels.id2label.items():       
                palette[label_id] = label.color  
                    
            output.putpalette(palette)    
            output.save(png_filename)       
            result_files.append(png_filename)                  
            
        return result_files              
    
    def format_results(self, results, imgfile_prefix, to_label_id=True, indices=None):   
        """Format the results into dir (standard format for ApolloScapes
        evaluation).    

        Args:
            results (list): Testing results of the dataset.    
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing  
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))        

        assert isinstance(results, list), 'results must be a list.'      
        assert isinstance(indices, list), 'indices must be a list.'       
        
        return self.results2img(results, imgfile_prefix, to_label_id, indices)          

    def evaluate(self, results, metric='mIoU', logger=None, imgfile_prefix=None):      
        """Evaluation in ApolloScapes/default protocol.       

        Args:   
            results (list): Testing results of the dataset.   
            metric (str | list[str]): Metrics to be evaluated.     
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for ApolloScapes evaluation only. It includes the file path and    
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with ApolloScapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of ApolloScapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: ApolloScapes/default metrics.    
        """       

        eval_results = {}          
        metrics = metric.copy() if isinstance(metric, list) else [metric]    
        if 'apolloscapes' in metrics:
            eval_results |= self._evaluate_apolloscapes(results, logger, imgfile_prefix)
            metrics.remove('apolloscapes')
        if len(metrics) > 0:
            eval_results.update(super(ApolloScapesDataset, self).evaluate(results, metrics, logger))

        return eval_results

    def _evaluate_apolloscapes(self, results, logger, imgfile_prefix):
        """Evaluation in ApolloScapes protocol.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file

        Returns:
            dict[str: float]: ApolloScapes evaluation results.
        """
        try:
            from dataset_api.lane_segmentation.evaluation import evalPixelLevelSemanticLabeling as ASEval  # noqa
        except ImportError as e:
            raise ImportError(
                'Please download "https://github.com/ApolloScapeAuto/dataset-api" '
                'to install ApolloScapesscripts first.'
            ) from e

        msg = 'Evaluating in ApolloScapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_dir = imgfile_prefix

        eval_results = {}
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        ASEval.args.evalInstLevelScore = True
        ASEval.args.predictionPath = osp.abspath(result_dir)
        ASEval.args.evalPixelAccuracy = True
        ASEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official ApolloScapesscripts,
        # **_bin.png is used
        for seg_map in mmcv.scandir(self.ann_dir, '_bin.png', recursive=True):
            seg_map_list.append(osp.join(self.ann_dir, seg_map))
            pred_list.append(ASEval.getPrediction(ASEval.args, seg_map))

        eval_results |= ASEval.evaluateImgLists(pred_list, seg_map_list, ASEval.args)

        return eval_results
