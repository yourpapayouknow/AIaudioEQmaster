import argparse
from .mastering import Config, master_audio

def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='Audio Mastering Tool')
    p.add_argument('input_file')
    p.add_argument('output_file')
    p.add_argument('--reference')
    p.add_argument('--genre')
    p.add_argument('--loudness', choices=['soft','dynamic','normal','loud'], default='normal')
    p.add_argument('--eq-profile', choices=['Neutral','Warm','Bright','Fusion'], default='Neutral')
    p.add_argument('--preview', action='store_true')
    return p

def main()->int:
    a=build_parser().parse_args()
    c=Config()
    c.loudness_option=a.loudness
    if a.reference: c.reference_file=a.reference
    elif a.genre: c.genre=a.genre
    master_audio(a.input_file, a.output_file, c, a.eq_profile, a.preview)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
