import json
from pathlib import Path
from autoeq.frequency_response import FrequencyResponse

DIR = Path().absolute()

fr =  FrequencyResponse(name="demo")

def measurement_rank(entry):
    order = [
        {'source': 'oratory1990', 'form': 'over-ear', 'rig': 'unknown'},
        {'source': 'crinacle', 'form': 'over-ear', 'rig': 'GRAS 43AG-7'},
        {'source': 'innerfidelity', 'form': 'over-ear', 'rig': 'unknown'},
        {'source': 'rtings', 'form': 'over-ear', 'rig': 'unknown'},
        {'source': 'headphonecom', 'form': 'over-ear', 'rig': 'unknown'},
        {'source': 'crinacle', 'form': 'over-ear', 'rig': 'EARS + 711'},

        {'source': 'crinacle', 'form': 'in-ear', 'rig': 'Bruel & Kjaer 4620'},
        {'source': 'oratory1990', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'crinacle', 'form': 'in-ear', 'rig': '711'},
        {'source': 'rtings', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'innerfidelity', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'headphonecom', 'form': 'in-ear', 'rig': 'unknown'},

        {'source': 'oratory1990', 'form': 'earbud', 'rig': 'unknown'},
        {'source': 'crinacle', 'form': 'earbud', 'rig': '711'},
        {'source': 'rtings', 'form': 'earbud', 'rig': 'unknown'},
        {'source': 'innerfidelity', 'form': 'earbud', 'rig': 'unknown'},
        {'source': 'headphonecom', 'form': 'earbud', 'rig': 'unknown'},
        {'source': 'freeryder05', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'Super Review', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'Harpo', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'Kazi', 'form': 'in-ear', 'rig': 'unknown'} ,
        {'source': 'Kazi', 'form': 'earbud', 'rig': 'unknown'} ,
        {'source': 'ToneDeafMonk', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'Jaytiss', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'Fahryst', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'Hi End Portable', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'RikudouGoku', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'kr0mka', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'kr0mka', 'form': 'earbud', 'rig': 'unknown'},
        {'source': "Ted's Squig Hoard", 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'Auriculares Argentina', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'Headphone.com Legacy', 'form': 'over-ear', 'rig': 'unknown'} ,
        {'source': 'Rtings', 'form': 'over-ear', 'rig': 'unknown'},
        {'source': 'Headphone.com Legacy', 'form': 'in-ear', 'rig': 'unknown'},
        {'source': 'Kuulokenurkka', 'form': 'over-ear', 'rig': 'unknown'} ,
           {'source': 'Filk', 'form': 'over-ear', 'rig': 'unknown'} ,
           {'source': 'Filk', 'form': 'in-ear', 'rig': 'unknown'},
           {'source': 'Super Review', 'form': 'over-ear', 'rig': 'unknown'} ,
            {'source': 'Auriculares Argentina', 'form': 'over-ear', 'rig': 'unknown'},
             {'source': 'kr0mka', 'form': 'over-ear', 'rig': 'unknown'},
              {'source': 'Rtings', 'form': 'earbud', 'rig': 'unknown'} ,
              {'source': 'Regan Cipher', 'form': 'earbud', 'rig': 'unknown'} ,
              {'source': 'Regan Cipher', 'form': 'over-ear', 'rig': 'unknown'} ,
              {'source': 'DHRME', 'form': 'in-ear', 'rig': 'unknown'},
               {'source': 'Super Review', 'form': 'earbud', 'rig': 'unknown'},
               {'source': 'Headphone.com Legacy', 'form': 'earbud', 'rig': 'unknown'},
               {'source': 'Bakkwatan', 'form': 'in-ear', 'rig': 'unknown'} ,
               {'source': 'oratory1990', 'form': 'in-ear', 'rig': 'unknown'},
               {'source': 'oratory1990', 'form': 'earbud', 'rig': 'unknown'},
               {'source': 'oratory1990', 'form': 'over-ear', 'rig': 'unknown'},
               {'source': 'rtings', 'form': 'over-ear', 'rig': 'unknown'},
               {'source': 'rtings', 'form': 'earbud', 'rig': 'unknown'},
               {'source': 'HypetheSonics', 'form': 'in-ear', 'rig': 'GRAS RA0045'},
               {'source': 'Innerfidelity', 'form': 'in-ear', 'rig': 'unknown'},
               {'source': 'Regan Cipher', 'form': 'in-ear', 'rig': 'unknown'},
               {'source': 'Innerfidelity', 'form': 'over-ear', 'rig': 'unknown'},
               {'source': 'Rtings', 'form': 'in-ear', 'rig': 'unknown'},
               {'source': 'HypetheSonics', 'form': 'in-ear', 'rig': 'Bruel & Kjaer 5128'},
                {'source': 'Innerfidelity', 'form': 'earbud', 'rig': 'unknown'},
                 {'source': 'HypetheSonics', 'form': 'over-ear', 'rig': 'unknown'},
                  {'source': 'HypetheSonics', 'form': 'earbud', 'rig': 'unknown'}
    ]
    return order.index({'source': entry['source'], 'form': entry['form'], 'rig': entry['rig']})


def write_entries_and_measurements():
    entries = dict()
    measurements = dict()
    for hp_path in DIR.parent.joinpath('measurements').glob('*/data/**/*.csv'):
        parts = hp_path.parts[hp_path.parts.index('data') + 1:]
        source = hp_path.parts[hp_path.parts.index('data') - 1]
        form = parts[0]
        rig = parts[1] if len(parts) == 3 else 'unknown'
        name = parts[-1].replace('.csv', '')
        if '(sample' in name or '(serial number' in name:
            # Skip individual samples
            continue
        if name not in entries:
            entries[name] = []
        if name not in measurements:
            measurements[name] = dict()
        if source not in measurements[name]:
            measurements[name][source] = dict()
        measurements[name][source][rig] = fr.read_csv(hp_path).to_dict()
        entries[name].append({
            'form': form, 'rig': rig, 'source': source
        })
    entries = {key: entries[key] for key in sorted(list(entries.keys()), key=lambda key: key)}
    for headphone in entries.keys():
        entries[headphone] = sorted(entries[headphone], key=lambda entry: measurement_rank(entry))
    with open(DIR.joinpath('data', 'measurements.json'), 'w', encoding='utf-8') as fh:
        json.dump(measurements, fh, ensure_ascii=False, indent=4)
    with open(DIR.joinpath('data', 'entries.json'), 'w', encoding='utf-8') as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=4)


def write_compensations():
    path = Path().resolve()
    compensations = [
        {
            'file': path.parent.joinpath('compensation', 'autoeq_in-ear.csv'),
            'label': 'AutoEq In-ear',
            'compatible': [
                {'source': 'crinacle', 'form': 'in-ear', 'rig': '711'},
                {'source': 'oratory1990', 'form': 'in-ear', 'rig': 'unknown'}
            ],
            'recommended': [
                {'source': 'crinacle', 'form': 'in-ear', 'rig': '711'},
                {'source': 'oratory1990', 'form': 'in-ear', 'rig': 'unknown'}
            ],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 9.5}
        },
        {
            'file': path.parent.joinpath('compensation', 'diffuse_field_5128_-1dBpoct.csv'),
            'label': 'Diffuse Field 5128 (-1 dB /oct)',
            'compatible': [{'source': 'crinacle', 'form': 'in-ear', 'rig': 'Bruel & Kjaer 4620'}],
            'recommended': [{'source': 'crinacle', 'form': 'in-ear', 'rig': 'Bruel & Kjaer 4620'}],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 0}
        },
        {
            'file': path.parent.joinpath('compensation', 'diffuse_field_gras_kemar.csv'),
            'label': 'Diffuse Field GRAS KEMAR',
            'compatible': [],
            'recommended': [],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 0}
        },
        {
            'file': path.parent.joinpath('compensation', 'diffuse_field_iso_11904-2.csv'),
            'label': 'Diffuse Field ISO 11904-2',
            'compatible': [],
            'recommended': [],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 0}
        },
        {
            'file': path.parent.joinpath('compensation', 'zero.csv'),
            'label': 'Flat',
            'compatible': [],
            'recommended': [],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 0}
        },
        {
            'file': path.parent.joinpath('compensation', 'harman_in-ear_2019v2_wo_bass.csv'),
            'label': 'Harman In-ear 2019',
            'compatible': [
                {'source': 'crinacle', 'form': 'in-ear', 'rig': '711'},
                {'source': 'oratory1990', 'form': 'in-ear', 'rig': 'unknown'}
            ],
            'recommended': [],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 9.5}
        },
        {
            'file': path.parent.joinpath('compensation', 'harman_over-ear_2018_wo_bass.csv'),
            'label': 'Harman Over-ear 2018',
            'compatible': [
                {'source': 'crinacle', 'form': 'over-ear', 'rig': 'GRAS 43AG-7'},
                {'source': 'oratory1990', 'form': 'over-ear', 'rig': 'unknown'}
            ],
            'recommended': [
                {'source': 'crinacle', 'form': 'over-ear', 'rig': 'GRAS 43AG-7'},
                {'source': 'oratory1990', 'form': 'over-ear', 'rig': 'unknown'}
            ],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 6}
        },
        {
            'file': path.parent.joinpath('compensation', 'headphonecom_autoeq_in-ear.csv'),
            'label': 'Headphone.com Legacy AutoEq In-ear',
            'compatible': [{'source': 'headphonecom', 'form': 'in-ear', 'rig': 'unknown'}],
            'recommended': [{'source': 'headphonecom', 'form': 'in-ear', 'rig': 'unknown'}],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 9.5}
        },
        {
            'file': path.parent.joinpath('compensation', 'headphonecom_harman_in-ear_2019v2_wo_bass.csv'),
            'label': 'Headphone.com Legacy Harman In-ear 2019',
            'compatible': [{'source': 'headphonecom', 'form': 'in-ear', 'rig': 'unknown'}],
            'recommended': [],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 9.5}
        },
        {
            'file': path.parent.joinpath('compensation', 'headphonecom_harman_over-ear_2018_wo_bass.csv'),
            'label': 'Headphone.com Legacy Harman Over-ear 2018',
            'compatible': [{'source': 'headphonecom', 'form': 'over-ear', 'rig': 'unknown'}],
            'recommended': [{'source': 'headphonecom', 'form': 'over-ear', 'rig': 'unknown'}],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 6}
        },
        {
            'file': path.parent.joinpath('compensation', '711_5128_delta.csv'),
            'label': '711/5128 Delta',
            'compatible': [
                {'source': 'crinacle', 'form': 'in-ear', 'rig': 'unknown'},
                {'source': 'oratory1990', 'form': 'in-ear', 'rig': 'unknown'}
            ],
            'recommended': [],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 0}
        },
        {
            'file': path.parent.joinpath('compensation', 'innerfidelity_autoeq_in-ear.csv'),
            'label': 'Innerfidelity AutoEq In-ear',
            'compatible': [{'source': 'innerfidelity', 'form': 'in-ear', 'rig': 'unknown'}],
            'recommended': [{'source': 'innerfidelity', 'form': 'in-ear', 'rig': 'unknown'}],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 9.5}
        },
        {
            'file': path.parent.joinpath('compensation', 'innerfidelity_harman_in-ear_2019v2_wo_bass.csv'),
            'label': 'Innerfidelity Harman In-ear 2019',
            'compatible': [{'source': 'innerfidelity', 'form': 'in-ear', 'rig': 'unknown'}],
            'recommended': [],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 9.5}
        },
        {
            'file': path.parent.joinpath('compensation', 'innerfidelity_harman_over-ear_2018_wo_bass.csv'),
            'label': 'Innerfidelity Harman Over-ear 2018',
            'compatible': [{'source': 'innerfidelity', 'form': 'over-ear', 'rig': 'unknown'}],
            'recommended': [{'source': 'innerfidelity', 'form': 'over-ear', 'rig': 'unknown'}],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 6}
        },
        {
            'file': path.parent.joinpath('compensation', 'rtings_autoeq_in-ear.csv'),
            'label': 'Rtings AutoEq In-ear',
            'compatible': [{'source': 'rtings', 'form': 'in-ear', 'rig': 'unknown'}],
            'recommended': [{'source': 'rtings', 'form': 'in-ear', 'rig': 'unknown'}],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 9.5}
        },
        {
            'file': path.parent.joinpath('compensation', 'rtings_harman_in-ear_2019v2_wo_bass.csv'),
            'label': 'Rtings Harman In-ear 2019',
            'compatible': [{'source': 'rtings', 'form': 'in-ear', 'rig': 'unknown'}],
            'recommended': [],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 9.5}
        },
        {
            'file': path.parent.joinpath('compensation', 'rtings_harman_over-ear_2018_wo_bass.csv'),
            'label': 'Rtings Harman Over-ear 2018',
            'compatible': [{'source': 'rtings', 'form': 'over-ear', 'rig': 'unknown'}],
            'recommended': [{'source': 'rtings', 'form': 'over-ear', 'rig': 'unknown'}],
            'bassBoost': {'fc': 105, 'q': 0.7, 'gain': 6}
        },
    ]
    for compensation in compensations:
        compensation['fr'] = fr.read_csv(compensation['file']).to_dict()
        del compensation['file']
    with open(DIR.joinpath('data', 'compensations.json'), 'w', encoding='utf-8') as fh:
        json.dump(compensations, fh, ensure_ascii=False, indent=4)


def main():
    write_entries_and_measurements()
    # write_compensations()


if __name__ == '__main__':
    main()