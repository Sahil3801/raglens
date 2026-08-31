"""One-time, deterministic packaging of the existing Phase 2 synthetic PDFs.

This is not the missing historical benchmark and is not representative quality
evidence. The manifest must be frozen BEFORE any evaluation results are observed.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'corpora' / 'synthetic_stations_v1'
SOURCE = ROOT.parents[1] / 'tmp' / 'runtime-verification'
NAMES = {s: f'raglens_runtime_20260831_{s}.pdf' for s in ['aurora', 'borealis']}
EXPECTED = {
    'aurora': 'cb3874b5a59366d1bc687b7ba47e77704cab9457e67da9573194ca0d9fedaf82',
    'borealis': '6f61c0ab99d6894e2b981ad86efc76edff2929dbc50bb4b6c06e489bf7b28bfa',
}


def evidence(station, page, quote):
    return {'source_file': NAMES[station], 'page': page - 1, 'contains': quote}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / 'manifest.json').exists():
        raise SystemExit('Manifest already frozen; create a NEW version to change it.')
    documents = []
    for station, filename in NAMES.items():
        data = (SOURCE / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == EXPECTED[station]
        shutil.copyfile(SOURCE / filename, DEST / filename)
        documents.append({'path': filename, 'sha256': EXPECTED[station], 'pages': 36})
    ac = evidence('aurora', 1, 'AMBER-731')
    bc = evidence('borealis', 1, 'COBALT-482')
    ai = evidence('aurora', 1, '17 days')
    bi = evidence('borealis', 1, '29 days')
    questions = []

    def add(qid, question, reference, groups, scope=None, answerable=True):
        questions.append({'id': qid, 'question': question, 'reference': reference,
                          'filter_filename': NAMES[scope] if scope else None,
                          'answerable': answerable, 'evidence_groups': groups})

    add('q01', 'What is the maintenance access code for Aurora station?',
        'The maintenance access code for Aurora station is AMBER-731.', [[ac]])
    add('q02', 'What is the maintenance access code for Borealis station?',
        'The maintenance access code for Borealis station is COBALT-482.', [[bc]])
    add('q03', 'How often is Aurora station inspected?',
        'Aurora station has an inspection interval of 17 days.', [[ai]], 'aurora')
    add('q04', 'How many days are between inspections at Borealis station?',
        'The inspection interval at Borealis station is 29 days.', [[bi]], 'borealis')
    add('q05', 'List the maintenance access codes for both Aurora and Borealis stations.',
        'Aurora: AMBER-731. Borealis: COBALT-482.', [[ac], [bc]])
    add('q06', 'Compare the inspection intervals at Aurora and Borealis stations.',
        'Aurora is inspected every 17 days; Borealis every 29 days.', [[ai], [bi]])
    add('q07', 'Which station is inspected more frequently, and by how many days do the intervals differ?',
        'Aurora is inspected more frequently. The intervals differ by 12 days (29 minus 17).', [[ai], [bi]])
    add('q08', 'Are the station maintenance access codes real credentials?',
        'No. Both handbooks describe their values as fictional identifiers for software verification, not real credentials.',
        [[evidence('aurora', 1, 'not real credentials')], [evidence('borealis', 1, 'not real credentials')]])
    add('q09', 'Which archive shelf does Procedure 27 cover at Aurora station?',
        'Procedure 27 covers archive shelf 27 at Aurora station.',
        [[evidence('aurora', 27, 'Procedure 27 covers archive shelf 27')]], 'aurora')
    add('q10', 'At Borealis station, what shelf is covered by Procedure 31?',
        'Procedure 31 covers archive shelf 31 at Borealis station.',
        [[evidence('borealis', 31, 'Procedure 31 covers archive shelf 31')]], 'borealis')
    add('q11', 'What should be checked and recorded before handling a storage box at Aurora station?',
        'Check its paper label and record the shelf number.',
        [[evidence('aurora', p, 'check its paper label and record the shelf number') for p in range(2, 37)]], 'aurora')
    add('q12', 'Where should completed archive housekeeping work at Borealis station be recorded?',
        'Use the inventory ledger to record completed housekeeping work.',
        [[evidence('borealis', p, 'Use the inventory ledger') for p in range(2, 37)]], 'borealis')
    refusal = 'I cannot answer this based on the provided document.'
    add('q13', 'What is the salary of the Aurora station manager?', refusal, [], 'aurora', False)
    add('q14', 'What will the weather be at Borealis station tomorrow?', refusal, [], 'borealis', False)
    add('q15', 'How many moons does Jupiter have?', refusal, [], None, False)
    add('q16', 'What is the maintenance access code for the third station, Cygnus?', refusal, [], None, False)
    manifest = {
        'schema_version': 1, 'benchmark_id': 'synthetic_stations_v1',
        'provenance': 'The unchanged two fictional Phase 2 runtime PDFs; original resume benchmark PDFs unavailable.',
        'limitations': ['Synthetic, tiny corpus with repetitive distractors; not representative of real PDF QA.',
                        'Questions share evidence and are not independent samples; no general population inference.',
                        'No historical scores are targets or accepted evidence.'],
        'seed': 20260831, 'documents': documents, 'questions': questions,
        'primary_population': '12 answerable questions; unanswerable items reported separately',
        'evidence_semantics': 'Each group is required; any listed chunk matching file/page/quote satisfies that group. Pages are zero-based. All alternatives enumerated.',
        'comparison': {'A': 'First 8 of the shared production MMR output, no CrossEncoder.',
                       'B': 'Unmodified production CrossEncoder rerank of that same MMR output, including threshold and 4-chunk floor, at most 8.',
                       'matched_count': 'Retrieval-only sensitivity analysis: compare first len(B) MMR chunks with B. No extra generation arm.'},
        'aggregation': 'Macro means with valid/failed/undefined counts; paired deltas use only pairs valid in both arms. Context Precision is not applicable for unanswerable questions.',
    }
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Frozen {len(documents)} PDFs and {len(questions)} questions at {DEST}')


if __name__ == '__main__':
    main()
