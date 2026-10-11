"""Render through the packaged DOCX renderer using a native Word PDF."""
import importlib.util
import os
from pathlib import Path
import shutil
import sys

runtime = Path('C:/Users/id116/.cache/codex-runtimes/codex-primary-runtime/dependencies')
skill = Path('C:/Users/id116/.codex/plugins/cache/openai-primary-runtime/documents/26.1009.11318/skills/documents')
os.environ['PATH'] = str(runtime / 'native/poppler/Library/bin') + os.pathsep + os.environ['PATH']
spec = importlib.util.spec_from_file_location('packaged_docx_renderer', skill / 'render_docx.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
pdf = Path(sys.argv[2]).resolve()

def native_pdf(input_path, user_profile, convert_tmp_dir, stem, verbose=False):
    destination = Path(convert_tmp_dir) / (Path(input_path).stem + '.pdf')
    shutil.copyfile(pdf, destination)
    return str(destination), 'Native Word export; runtime.json confirms no bundled LibreOffice.'

renderer.convert_to_pdf = native_pdf
sys.argv = [str(skill / 'render_docx.py'), sys.argv[1], '--output_dir', sys.argv[3], '--dpi', '144']
renderer.main()
