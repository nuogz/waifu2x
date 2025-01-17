# @nuogz/waifu2x
![Version](https://img.shields.io/github/package-json/v/nuogz/waifu2x?style=flat-square)
[![License](https://img.shields.io/github/license/nuogz/waifu2x?style=flat-square)](https://www.gnu.org/licenses/lgpl-3.0-standalone.html)

Waifu2x的Node.js调用版本，根据官网无限制版本的代码改写而来，使用官方提供的预训练好的onnx模型  
Waifu2x in Node.js.  
Adapted from the code on [the official website unlimited version](https://unlimited.waifu2x.net/).  
Will use pre-trained onnx models from [waifu2x official repository](https://github.com/nagadomi/nunif).


> **警告：  
尽管该库的代码目前处于公开状态，但该库目前仅为我个人使用。所有功能与设计均基于我的日常使用需求。  
由于我个人精力和时间有限，无法对该库的代码和文档作出任何质量保证，也无法对过期的版本提供任何支持。而对于最新版本，也只能最低限度的慢速的支持。  
因此不推荐任何用户使用该库，除非对其代码有充分的了解和信心。**

> **Warning:  
Although the code for this library is currently in a public state, the library is currently for my personal use only. All functionality and design is based on my daily usage requirements.  
Due to my limited personal energy and time, I cannot provide any quality assurance for the code and documentation of this library, nor can I provide any support for outdated versions. I can't provide any support for outdated versions, and I can only support the latest version at a slow minimum.  
Therefore, I do not recommend any user to use this library unless they have full knowledge and confidence in its code.**

> Most of the English content in this document was generated by machine translation (if available).


## 下载模型
由于官方提供的模型文件太大，本仓库不会内置模型文件。  
需从[waifu2x官方发布](https://github.com/nagadomi/nunif/releases)中下载`waifu2x_onnx_models_*.zip`文件，并解压到你觉得合适存放的位置，最后通过参数指定模型文件夹位置
