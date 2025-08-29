from sf2utils.sf2parse import Sf2File

def print_sorted_presets(sf2_path):
    presets_info = []

    with open(sf2_path, 'rb') as f:
        sf2 = Sf2File(f)

        for preset in sf2.presets:
            try:
                # 尝试直接读取
                name = getattr(preset, 'name', '???').strip('\x00')
                bank = getattr(preset, 'bank', None)
                program = getattr(preset, 'preset', None)

                # 如果获取不到，再尝试从子属性中取
                if bank is None or program is None:
                    for attr in dir(preset):
                        attr_value = getattr(preset, attr)
                        if hasattr(attr_value, 'bank') and hasattr(attr_value, 'preset'):
                            bank = attr_value.bank
                            program = attr_value.preset
                            name = getattr(attr_value, 'name', name).strip('\x00')
                            break

                # 收集有效结果
                if bank is not None and program is not None:
                    presets_info.append((program, bank, name))
            except Exception as e:
                print(f"Error reading preset: {e}")

    # 按 program 升序排序（若需要按 bank 再 program，改为 sorted(..., key=lambda x: (x[1], x[0]))）
    presets_info.sort(key=lambda x: x[0])

    # 打印结果
    print(f"{'Program':<8} {'Bank':<6} {'Preset Name'}")
    print("-" * 40)
    for program, bank, name in presets_info:
        print(f"{program:<8} {bank:<6} {name}")

# DEFAULT_SOUND_FONT = '/data2/suhongju/research/music-generation/sound_file/CrisisGeneralMidi3.01.sf2'
# DEFAULT_SOUND_FONT = '~/.fluidsynth/default_sound_font.sf2'

# 替换为你的 sf2 文件路径
sf2_path = "/data2/suhongju/research/music-generation/sound_file/CrisisGeneralMidi3.01.sf2"
print_sorted_presets(sf2_path)