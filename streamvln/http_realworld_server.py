import argparse
import numpy as np
import io
import json
import threading
import time
import torch
import traceback
import sys
import os
import transformers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify, Response
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from streamvln.streamvln_agent import VLNEvaluator
from model.stream_video_vln import StreamVLNForCausalLM

app = Flask(__name__)
action_seq = np.zeros(4)
idx = 0
terminate = False
total_generate_time = 0.0
start_time = time.time()
output_dir = ''

# The instruction used when a request does not carry one (keeps the stock go2 client,
# which posts only {"reset": bool}, working unchanged).
DEFAULT_INSTRUCTION = "Walk forward and immediately stop when you exit the room."
# Newest annotated frame (JPEG bytes) + last inference summary, for the /debug view.
latest_annotated = None
last_info = {}
# Held for the lifetime of a session: StreamVLN only re-reads the instruction when it
# rebuilds the prompt (every memory reset), so it must outlive the request that set it.
instruction = DEFAULT_INSTRUCTION
def annotate_image(idx, image, start_time, total_generate_time, llm_output, output_dir):
    image = Image.fromarray(image)#.save(f'rgb_{idx}.png')
    draw = ImageDraw.Draw(image)
    font_size = 20
    font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    text_content = []
    text_content.append(f"Frame    Id  : {idx}")
    text_content.append(f"Running  time: {time.time() - start_time:.2f} s")
    text_content.append(f"Generate time: {total_generate_time:.2f} s")
    text_content.append(f"Actions      : {llm_output}" )
    max_width = 0
    total_height = 0
    for line in text_content:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = 26
        max_width = max(max_width, text_width)
        total_height += text_height

    padding = 10
    box_x, box_y = 10, 10
    box_width = max_width + 2 * padding
    box_height = total_height + 2 * padding

    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

    text_color = 'white'
    y_position = box_y + padding
    
    for line in text_content:
        draw.text((box_x + padding, y_position), line, fill=text_color, font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        text_height = 26
        y_position += text_height

    image.save(f'{output_dir}/rgb_{idx}_annotated.png')

    # Keep the newest annotated frame in memory too, so /debug can serve a live view
    # without anyone having to shell into the box and open PNGs.
    global latest_annotated
    buf = io.BytesIO()
    image.save(buf, format='jpeg', quality=85)
    latest_annotated = buf.getvalue()

# The evaluator holds ONE global KV-cache session. Two clients interleaving requests
# silently corrupts it -- SDPA then dies with "Expected key.size(1) == value.size(1)" and
# every later request fails until the process is restarted. Serialize access, and if
# inference does throw, drop the cache so the next request starts clean instead of
# inheriting the wreckage.
_session_lock = threading.Lock()
SESSION_LOCK_TIMEOUT_S = 60.0


@app.route("/eval_vln", methods=['POST'])
def eval_vln():
    if not _session_lock.acquire(timeout=SESSION_LOCK_TIMEOUT_S):
        return jsonify({'error': 'busy: another client holds the session'}), 429
    try:
        return _eval_vln_impl()
    except Exception as exc:
        traceback.print_exc()
        try:
            evaluator.reset_memory()
        except Exception:
            print("session reset FAILED after error; restart the server")
        return jsonify({'error': f'{type(exc).__name__}: {exc}',
                        'session_reset': True}), 503
    finally:
        _session_lock.release()


def _eval_vln_impl():
    global action_seq, idx, terminate, total_generate_time, output_dir, start_time, instruction

    image_file = request.files['image']
    json_data = request.form['json']
    data = json.loads(json_data)
    
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)[...,::-1]

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Instruction comes from the request when present, otherwise the session keeps the
    # last one it was given. A reset with no instruction falls back to the default.
    req_instruction = (data.get('instruction') or '').strip()
    policy_init = bool(data.get('reset', False))
    if req_instruction:
        instruction = req_instruction
    elif policy_init:
        instruction = DEFAULT_INSTRUCTION

    if policy_init:
        start_time = time.time()
        total_generate_time = 0.0
        terminate = False
        idx = 0
        output_dir = 'runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print(f"init reset model!!! instruction: {instruction!r}")
        evaluator.reset_memory()
    
    idx += 1
    
    if terminate:
        print("!!!!!!!!!!!!!!!!!task finish!!!!!!!!!!!!!!!!!!!!!")
        return jsonify({'action': [0]})
    
    for i in range(4):
        t1 = time.time()
        depth = np.zeros((image.shape[0], image.shape[1], 1))
        return_action, generate_time, return_llm_output = evaluator.step(0,
                                        image,
                                        #depth,
                                        #camera_pose,
                                        instruction,
                                        run_model=(evaluator.step_id % 4 == 0))
        llm_output = return_llm_output if return_llm_output is not None else llm_output
        print(f"one evalute cost {time.time() - t1}")
        # total_generate_time += generate_time
        
        if generate_time > 0:
            total_generate_time = generate_time
        action_seq = action_seq if return_action is None else return_action
        if 0 in action_seq:
            terminate = True     
        evaluator.step_id += 1
        
    str_action = [str(i) for i in action_seq]
    str_action = ''.join(str_action)
    str_action = str_action.replace('1', '↑')  # 前箭头
    str_action = str_action.replace('2', '←')  # 左箭头
    str_action = str_action.replace('3', '→')  # 右箭头
    str_action = str_action.replace('0', 'STOP')  # 停止
    if idx > 1 and total_generate_time > 0.5:
        total_generate_time -= 0.3

    annotate_image(idx, image, start_time, total_generate_time, str_action, output_dir)
    
    if len(action_seq) == 0:
        print("!!!!!!!!!!!!!!!!!task finish!!!!!!!!!!!!!!!!!!!!!")
        return jsonify({'action': [0]})
    
    last_info.update({'frame': idx, 'action': list(action_seq), 'glyphs': str_action,
                      'instruction': instruction, 'generate_time': total_generate_time,
                      'elapsed': time.time() - start_time})

    # 'action' keeps the stock contract; the extra fields let a client verify which
    # instruction the session is actually running under and track inference latency.
    return jsonify({'action': action_seq,
                    'instruction': instruction,
                    'generate_time': total_generate_time})


@app.route("/debug")
def debug_view():
    """Live view of what the model is actually seeing and deciding.

    Auto-refreshing page showing the newest annotated frame plus the last inference
    summary. Handy for watching a run from a laptop instead of tailing container logs.
    """
    return f"""<!doctype html><meta charset="utf-8"><title>StreamVLN debug</title>
<meta http-equiv="refresh" content="1">
<style>
 body{{background:#111;color:#eee;font:14px/1.5 monospace;margin:0;padding:16px}}
 img{{max-width:100%;border:1px solid #333;image-rendering:pixelated}}
 table{{border-collapse:collapse;margin-top:12px}} td{{padding:2px 14px 2px 0}}
 .k{{color:#888}} .none{{color:#c66}}
</style>
<h2 style="margin:0 0 12px">StreamVLN &mdash; live debug</h2>
{'<img src="/debug/frame.jpg?t=' + str(time.time()) + '">'
 if latest_annotated else '<p class="none">no frame yet &mdash; waiting for the first request</p>'}
<table>
{''.join(f'<tr><td class="k">{k}</td><td>{v}</td></tr>' for k, v in last_info.items())}
<tr><td class="k">terminated</td><td>{terminate}</td></tr>
<tr><td class="k">output_dir</td><td>{output_dir}</td></tr>
</table>"""


@app.route("/debug/frame.jpg")
def debug_frame():
    if latest_annotated is None:
        return Response(status=404)
    return Response(latest_annotated, mimetype='image/jpeg')


@app.route("/status")
def status():
    """Machine-readable sibling of /debug, for scripts and the ROS node."""
    return jsonify({'ready': True, 'terminated': terminate,
                    'instruction': instruction, 'busy': _session_lock.locked(),
                    **last_info})
    
if __name__ == '__main__':
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/pjlab/yq_ws/StreamVLN/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for testing')
    
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path,
                                                        model_max_length=args.model_max_length,
                                                        padding_side="right")
    
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
                args.model_path,
                attn_implementation=os.environ.get("STREAMVLN_ATTN", "sdpa"),  # flash_attention_2 needs a CUDA build; sdpa works everywhere
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=False,
                )
    model.model.num_history = args.num_history
    model.reset(1)
    model.requires_grad_(False)
    model.to(args.device)
    model.eval()
    
    
    vln_sensor_config = {
        "rgb_height" : 1.25, 
        "camera_intrinsic" : np.array([[192.        ,   0.        , 191.42857143,   0.        ],
            [  0.        , 192.        , 191.42857143,   0.        ],
            [  0.        ,   0.        ,   1.        ,   0.        ],
            [  0.        ,   0.        ,   0.        ,   1.        ]]),
    }
    
    evaluator = VLNEvaluator(
        vln_sensor_config,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )
    
    
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=True)
    app.run(host='0.0.0.0', port=
            5801)
