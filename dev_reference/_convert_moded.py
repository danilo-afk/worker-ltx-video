import json, sys

SRC="/Users/danilo/Documents/platform_k/LTX-2.3_ICLoRA_Ingredients_Single_Stage_Moded.json"
d=json.load(open(SRC))
nodes={n['id']:n for n in d['nodes']}
# links: id -> (from_node, from_slot, to_node, to_slot)
L={l[0]:(l[1],l[2],l[3],l[4]) for l in d['links']}

# schema de widgets (nomes ordenados) por tipo — só os que sobrevivem
WIDGETS={
 'CheckpointLoaderSimple':['ckpt_name'],
 'CLIPTextEncode':['text'],
 'LoraLoaderModelOnly':['lora_name','strength_model'],
 'LTXICLoRALoaderModelOnly':['lora_name','strength_model'],
 'LTXAddVideoICLoRAGuide':['frame_idx','strength','latent_downscale_factor','crop','use_tiled_encode','tile_size','tile_overlap'],
 'KSamplerSelect':['sampler_name'],
 'RandomNoise':['noise_seed','control_after_generate'],
 'CFGGuider':['cfg'],
 'LTXVConditioning':['frame_rate'],
 'EmptyLTXVLatentVideo':['width','height','length','batch_size'],
 'LTXVTiledVAEDecode':['horizontal_tiles','vertical_tiles','overlap','last_frame_fix','working_device','working_dtype'],
 'LTXVEmptyLatentAudio':['frames_number','frame_rate','batch_size'],
 'LTXVPreprocess':['img_compression'],
 'LTXVImgToVideoConditionOnly':['strength','bypass'],
 'LTXAVTextEncoderLoader':['text_encoder','ckpt_name','device'],
 'LTXVAudioVAELoader':['ckpt_name'],
 'ManualSigmas':['sigmas'],
 'PrimitiveInt':['value','control_after_generate'],
 'PrimitiveFloat':['value'],
 'PrimitiveBoolean':['value'],
 'LoadImage':['image','upload'],
 'RepeatImageBatch':['amount'],
 'CreateVideo':['fps'],
 'LTXFloatToInt':[],
 'SamplerCustomAdvanced':[],'LTXVConcatAVLatent':[],'LTXVAudioVAEDecode':[],
 'LTXVSeparateAVLatent':[],'LTXVCropGuides':[],'GetVideoComponents':[],
}
DROP={'TextGenerate','ShowText|pysssss','Note','ResizeImageMaskNode','GetImageSize'}
FP8={'ltx2\\ltx-2.3-22b-dev.safetensors':'ltx-2.3-22b-dev-fp8.safetensors',
     'gemma_3_12B_it_fp8_scaled.safetensors':'gemma_3_12B_it.safetensors',
     'ltx2\\ltx-2.3-22b-distilled-lora-384-1.1.safetensors':'ltx-2.3-22b-distilled-lora-384-1.1.safetensors',
     'LTX2_3\\ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors':'ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors'}

def resolve_link(link_id):
    fn,fs,_,_=L[link_id]; return [str(fn),fs]

api={}
warn=[]
for nid,n in nodes.items():
    t=n['type']
    if t in DROP: continue
    inputs={}
    # 1) inputs conectados (links)
    linked=set()
    for i in n.get('inputs',[]):
        if i.get('link') is not None:
            src_node=L[i['link']][0]
            if nodes[src_node]['type'] in DROP:  # link vem de nó removido -> vira widget/placeholder
                continue
            inputs[i['name']]=resolve_link(i['link'])
            linked.add(i['name'])
    # 2) widgets
    wnames=WIDGETS.get(t)
    wv=n.get('widgets_values')
    if wnames is None and t not in ('VHS_VideoCombine',):
        warn.append(f'sem schema: {t}')
    if isinstance(wv,list) and wnames:
        for i,name in enumerate(wnames):
            if i>=len(wv): break
            if name in linked: continue  # link vence
            val=wv[i]
            if name in ('ckpt_name','lora_name','text_encoder'):
                val=FP8.get(val,val.split('\\')[-1])
            inputs[name]=val
    api[str(nid)]={'class_type':t,'inputs':inputs}

json.dump(api, open('ltx23_iclora_draft.json','w'), indent=1)
print('nós API:', len(api))
print('warnings:', set(warn))
# pontos de injeção
for nid,n in api.items():
    if n['class_type']=='CLIPTextEncode': print('CLIPTextEncode', nid, 'text=', repr(n['inputs'].get('text'))[:30])
    if n['class_type']=='PrimitiveInt': print('PrimitiveInt(length?)', nid, n['inputs'])
    if n['class_type']=='LoadImage': print('LoadImage', nid, n['inputs'])
    if n['class_type']=='LTXAddVideoICLoRAGuide': print('Guide', nid, {k:v for k,v in n['inputs'].items()})
    if n['class_type']=='EmptyLTXVLatentVideo': print('EmptyLatent', nid, n['inputs'])
