import json
d=json.load(open('iclora_ing.json'))
nodes={n['id']:n for n in d['nodes']}
L={l[0]:(l[1],l[2],l[3],l[4]) for l in d['links']}
WIDGETS={
 'CheckpointLoaderSimple':['ckpt_name'],'CLIPTextEncode':['text'],
 'LoraLoaderModelOnly':['lora_name','strength_model'],'LTXICLoRALoaderModelOnly':['lora_name','strength_model'],
 'LTXAddVideoICLoRAGuide':['frame_idx','strength','latent_downscale_factor','crop','use_tiled_encode','tile_size','tile_overlap'],
 'KSamplerSelect':['sampler_name'],'RandomNoise':['noise_seed','control_after_generate'],'CFGGuider':['cfg'],
 'LTXVConditioning':['frame_rate'],'EmptyLTXVLatentVideo':['width','height','length','batch_size'],
 'LTXVTiledVAEDecode':['horizontal_tiles','vertical_tiles','overlap','last_frame_fix','working_device','working_dtype'],
 'LTXVEmptyLatentAudio':['frames_number','frame_rate','batch_size'],'LTXVPreprocess':['img_compression'],
 'LTXVImgToVideoConditionOnly':['strength','bypass'],'LTXAVTextEncoderLoader':['text_encoder','ckpt_name','device'],
 'LTXVAudioVAELoader':['ckpt_name'],'ManualSigmas':['sigmas'],
 'PrimitiveInt':['value','control_after_generate'],'PrimitiveFloat':['value'],'PrimitiveBoolean':['value'],'PrimitiveString':['value'],
 'LoadImage':['image','upload'],'RepeatImageBatch':['amount'],'CreateVideo':['fps'],'SaveVideo':['filename_prefix','format','codec'],
 'LTXFloatToInt':[],'SamplerCustomAdvanced':[],'LTXVConcatAVLatent':[],'LTXVAudioVAEDecode':[],
 'LTXVSeparateAVLatent':[],'LTXVCropGuides':[],'GetVideoComponents':[],'GetImageSize':[],
 'ResizeImageMaskNode':['keep_proportion','resize_value','upscale_method'],
 'GemmaAPITextEncode':['prompt','model','api_key'],
}
FP8={'ltx2\\ltx-2.3-22b-dev.safetensors':'ltx-2.3-22b-dev-fp8.safetensors','ltx-2.3-22b-dev.safetensors':'ltx-2.3-22b-dev-fp8.safetensors',
 'gemma_3_12B_it_fp8_scaled.safetensors':'gemma_3_12B_it.safetensors',
 'LTX2_3\\ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors':'ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors',
 'ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors':'ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors',
 'ltx2\\ltx-2.3-22b-distilled-lora-384-1.1.safetensors':'ltx-2.3-22b-distilled-lora-384-1.1.safetensors'}
def fix(v):
    if isinstance(v,str):
        base=v.split('\\')[-1]
        return FP8.get(v, FP8.get(base, base))
    return v
api={}
for nid,n in nodes.items():
    t=n['type']; inputs={}; linked=set()
    for i in n.get('inputs',[]):
        if i.get('link') is not None and i['link'] in L:
            fn,fs,_,_=L[i['link']]
            if fn in nodes: inputs[i['name']]=[str(fn),fs]; linked.add(i['name'])
    wn=WIDGETS.get(t,[]); wv=n.get('widgets_values')
    if isinstance(wv,list):
        for idx,name in enumerate(wn):
            if idx>=len(wv) or name in linked: continue
            val=wv[idx]
            if name in ('ckpt_name','lora_name','text_encoder'): val=fix(val)
            inputs[name]=val
    api[str(nid)]={'class_type':t,'inputs':inputs}
json.dump(api,open('iclora_api.json','w'),indent=1)
print('nós API:',len(api))
# pontos de injeção
for nid,n in api.items():
    if n['class_type']=='PrimitiveString': print('PrimitiveString(prompt?)',nid,repr(str(n['inputs'].get('value'))[:40]))
    if n['class_type']=='LoadImage': print('LoadImage',nid,n['inputs'].get('image'))
    if n['class_type']=='LTXICLoRALoaderModelOnly': print('ICLoRALoader',nid,n['inputs'])
    if n['class_type']=='LTXAddVideoICLoRAGuide': print('Guide',nid,{k:v for k,v in n['inputs'].items() if not isinstance(v,list)})
    if n['class_type']=='EmptyLTXVLatentVideo': print('EmptyLatent',nid,n['inputs'])
    if n['class_type']=='CheckpointLoaderSimple': print('Checkpoint',nid,n['inputs'])
    if n['class_type']=='GemmaAPITextEncode': print('GemmaAPI!',nid,n['inputs'])
