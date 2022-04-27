# @ FileName:logger.py
# @ Author: Yujun
# @ Time: 21-8-2 下午2:34
import sys
import time
import functools
from os.path import join
from datetime import datetime
import logging
import logging.handlers
from utils import tools
from tensorboardX import SummaryWriter

logger = {'pblog':None, 'tblog':None}

def get_logger(args, is_colorful=False):
    global logger
    if logger['pblog'] == None:
        logger['pblog'] = logging.getLogger(args.log_name)
        
        logger['pblog'].propagate = False    # 万分关键！！！禁止logger propagate，防止重复显示

        cmd_fmt = '[%(asctime)s] @%(name)s %(levelname)-8s%(message)s'
        cmd_datefmt = '%Y-%m-%d %H:%M:%S'
        if args.cmd == 'train':
            name = tools.gen_name(args.desc,'.log')
            log_path = join(args.log_dir, name)
            file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter(cmd_fmt, cmd_datefmt)
            file_handler.setFormatter(formatter)
            logger['pblog'].addHandler(file_handler)

        if args.show_log:
            console_handler = logging.StreamHandler()
            if is_colorful:
                color_formatter = ColoredFormatter(cmd_fmt, cmd_datefmt)
                console_handler.setFormatter(color_formatter)
            else:
                console_handler.setFormatter(formatter)

            logger['pblog'].addHandler(console_handler)
        logger['pblog'].setLevel(logging.DEBUG)
    
    if logger['tblog'] == None and args.cmd=='train':
        logger['tblog'] = SummaryWriter(join(args.tb_dir, args.desc))

    return logger

class ColoredFormatter(logging.Formatter):
    """自定义 ColoredFormatter 类"""
    def __init__(self, fmt=None, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        # Color escape string
        # 以下的字符串能够设置输出的颜色，其中\033是 ESC 的8进制表示形式
        COLOR_RED = '\033[1;31m'
        COLOR_GREEN = '\033[1;32m'
        COLOR_YELLOW = '\033[1;33m'
        COLOR_BLUE = '\033[1;34m'
        COLOR_PURPLE = '\033[1;35m'
        COLOR_CYAN = '\033[1;36m'
        COLOR_GRAY = '\033[1;37m'
        COLOR_WHITE = '\033[1;38m'
        COLOR_RESET = '\033[1;0m'
        # Define log color
        LOG_COLORS = {
            'DEBUG': COLOR_BLUE + '%s' + COLOR_RESET,
            'INFO': COLOR_GREEN + '%s' + COLOR_RESET,
            'WARNING': COLOR_YELLOW + '%s' + COLOR_RESET,
            'ERROR': COLOR_RED + '%s' + COLOR_RESET,
            'CRITICAL': COLOR_RED + '%s' + COLOR_RESET,
            'EXCEPTION': COLOR_RED + '%s' + COLOR_RESET,
        }
        level_name = record.levelname
        msg = logging.Formatter.format(self, record)
        # 此处的get方法第二个参数用作在字典中查不到key时返回，后面的 % 是格式化输出
        return LOG_COLORS.get(level_name, '%s') % msg   
