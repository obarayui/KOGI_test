!pip uninstall -y t5maru
!pip install -q git+https://github.com/KuramitsuLab/t5maru.git

import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

import time
import sys
import random
import IPython
from google.colab import output

class T5ModelWrapper:
    def __init__(self, model_path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, text):
        inputs = self.tokenizer.batch_encode_plus(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs.input_ids.to(torch.long)
        attention_mask = inputs.attention_mask.to(torch.long)
        
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_text[0]

# インターネットでダウンロードできる時に実行
from huggingface_hub import snapshot_download
download_path = snapshot_download(repo_id="parasora/KOGI")

# モデルのパスを指定してラッパーオブジェクトを作成
model_path = "parasora/KOGI"
t5_wrapper = T5ModelWrapper(model_path)

n = 0 
def chat(text, **kw):  #チャット用の関数（ここを書き換える）
  global n
  n += 1
  return 'ほ' * n

# アイコンの指定
#BOT_ICON = 'https://cdn.pixabay.com/photo/2021/07/24/23/15/corgi-6490554_960_720.png'
BOT_ICON = 'https://drive.google.com/uc?id=10EjF4vLI2UZfngUY3ELCI0O2cFWD3Q7Z'
YOUR_ICON = 'https://1.bp.blogspot.com/-ZOg0qAG4ewU/Xub_uw6q0DI/AAAAAAABZio/MshyuVBpHUgaOKJtL47LmVkCf5Vge6MQQCNcBGAsYHQ/s1600/pose_pien_uruuru_woman.png'

# チャットを起動
def run_chat(chat = chat, start='KOGIだよ！なんでも話しかけてね！', **kw):    #チャットを起動

  # botの表示
  def display_bot(bot_text):
    with output.redirect_to_element('#output'):
      #表示名
      bot_name = kw.get('bot_name', 'KOGI')
      bot_icon = kw.get('bot_icon', BOT_ICON)
      display(IPython.display.HTML(f'''
      <div class="sb-box">
        <div class="icon-img icon-img-left">
            <img src="{bot_icon}" width="60px">
        </div><!-- /.icon-img icon-img-left -->
        <div class="icon-name icon-name-left">{bot_name}</div>
        <div class="sb-side sb-side-left">
            <div class="sb-txt sb-txt-left">
              {bot_text}
            </div><!-- /.sb-txt sb-txt-left -->
        </div><!-- /.sb-side sb-side-left -->
    </div><!-- /.sb-box -->
      '''))

  # ユーザの表示
  def display_you(your_text):
    with output.redirect_to_element('#output'):
      your_name = kw.get('your_name', 'あなた')
      your_icon = kw.get('your_icon', YOUR_ICON)

      display(IPython.display.HTML(f'''
      <div class="sb-box">
        <div class="icon-img icon-img-right">
            <img src="{your_icon}" width="60px">
        </div><!-- /.icon-img icon-img-right -->
        <div class="icon-name icon-name-right">{your_name}</div>
        <div class="sb-side sb-side-right">
            <div class="sb-txt sb-txt-right">
              {your_text}
            </div><!-- /.sb-txt sb-txt-right -->
        </div><!-- /.sb-side sb-side-right -->
      </div><!-- /.sb-box -->
      '''))

  display(IPython.display.HTML('''
      <style>
        /* 全体 */
        .sb-box {
            position: relative;
            overflow: hidden;
        }
        /* アイコン画像 */
        .icon-img {
            position: absolute;
            overflow: hidden;
            top: 0;
            width: 100px;
            height: 100px;
        }
        /* アイコン画像（左） */
        .icon-img-left {
            left: 0;
        }
        /* アイコン画像（右） */
        .icon-img-right {
            right: 0;
        }
        /* アイコン画像 */
        .icon-img img {
            border-radius: 50%;
            border: 2px solid #eee;
        }
        /* アイコンネーム */
        .icon-name {
            position: absolute;
            width: 80px;
            text-align: center;
            top: 83px;
            color: #6C584C;
            font-size: 15px;
        }
        /* アイコンネーム（左） */
        .icon-name-left {
            left: 0;
        }
        /* アイコンネーム（右） */
        .icon-name-right {
            right: 0;
        }
        /* 吹き出し */
        .sb-side {
            position: relative;
            float: left;
            margin: 0 105px 40px 105px;
        }
        .sb-side-right {
            float: right;
        }
        /* 吹き出し内のテキスト */
        .sb-txt {
            position: relative;
            border: 2px solid  #F0EAD2;
            border-radius: 6px;
            background:  #F0EAD2;
            color: #6C584C;
            font-size: 15px;
            line-height: 1.7;
            padding: 18px;
        }
        .sb-txt>p:last-of-type {
            padding-bottom: 0;
            margin-bottom: 0;
        }
        /* 吹き出しの三角 */
        .sb-txt:before {
            content: "";
            position: absolute;
            border-style: solid;
            top: 16px;
            z-index: 3;
        }
        .sb-txt:after {
            content: "";
            position: absolute;
            border-style: solid;
            top: 15px;
            z-index: 2;
        }
        /* 吹き出しの三角（左） */
        .sb-txt-left:before {
            left: -7px;
            border-width: 7px 10px 7px 0;
            border-color: transparent #F0EAD2 transparent transparent;
        }
        .sb-txt-left:after {
            left: -10px;
            border-width: 8px 10px 8px 0;
            border-color: transparent #F0EAD2 transparent transparent;
        }
        /* 吹き出しの三角（右） */
        .sb-txt-right:before {
            right: -7px;
            border-width: 7px 0 7px 10px;
            border-color: transparent transparent transparent #F0EAD2;
        }
        .sb-txt-right:after {
            right: -10px;
            border-width: 8px 0 8px 10px;
            border-color: transparent transparent transparent #F0EAD2;
        }
        /* 767px（iPad）以下 */
        @media (max-width: 767px) {
            .icon-img {
                width: 60px;
                height: 60px;
            }
            /* アイコンネーム */
            .icon-name {
                width: 60px;
                top: 62px;
                font-size: 9px;
            }
            /* 吹き出し（左） */
            .sb-side-left {
                margin: 0 0 30px 78px;
                /* 吹き出し（左）の上下左右の余白を狭く */
            }
            /* 吹き出し（右） */
            .sb-side-right {
                margin: 0 78px 30px 0;
                /* 吹き出し（右）の上下左右の余白を狭く */
            }
            /* 吹き出し内のテキスト */
            .sb-txt {
                padding: 12px;
                /* 吹き出し内の上下左右の余白を-6px */
            }
        }
    </style>
      <script>
        var inputPane = document.getElementById('input');
        inputPane.addEventListener('keydown', (e) => {
          if(e.keyCode == 13) {
            google.colab.kernel.invokeFunction('notebook.Convert', [inputPane.value], {});
            inputPane.value=''
          }
        });
      </script>
    <div id='output' style='background: #ADC178;'></div>
    <div style='text-align: right'><textarea id='input' style='width: 100%; background: #eee;'></textarea></div>
      '''))

  def convert(your_text):
    display_you(your_text)
    bot_text = chat(your_text, **kw)
    time.sleep(random.randint(0,4))
    display_bot(bot_text)

  output.register_callback('notebook.Convert', convert)
  if start is not None:
    display_bot(start)

def kogioutput(input_text):
    global frame
    
    if frame['userinput'] == None:
        frame['userinput']=input_text.strip()
        output_text = t5_wrapper.generate(frame['userinput'])
        frame['kogioutput'] = output_text
        frame['userinput'] = None #ここを追加しました（小原）
        return output_text

def start():
  run_chat(chat=kogioutput)  

frame={'userinput':None, 'kogioutput':None}
