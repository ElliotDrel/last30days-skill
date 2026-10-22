from dataclasses import asdict
import threading
from unittest.mock import Mock, patch
import pytest
from lib import entity_extract, pipeline, planner, rerank, schema


def plan():
    return schema.QueryPlan('factual', 'strict_recent', 'story', 'Alex Rowan',
        [schema.SubQuery('primary', 'Alex Rowan Northstar Labs', 'Alex Rowan at Northstar Labs', ['x'])],
        {'x': 1.0}, person_identity_anchors=['Northstar Labs', 'Cedar Ventures'])


def item(text, author='disclosurebot', source='x'):
    return schema.SourceItem('i', source, text, text, 'https://example.invalid/post',
                             author=author, relevance_hint=1.0)


def candidate(evidence):
    return schema.Candidate('c', 'i', evidence.source, evidence.title, evidence.url,
        evidence.body, ['primary'], {'primary:x': 1}, 1.0, 100, 100000, 1.0, 1.0,
        source_items=[evidence])


@pytest.mark.parametrize('text,author,source,expected', [
    ('Alex Rowan resigns as Alloy Works secretary', 'disclosurebot', 'x', False),
    ('Alex Rowan joins NORTHSTAR LABS', 'reporter', 'x', True),
    ('Alex Rowan joins Cedar Ventures', 'reporter', 'x', True),
    ('Northstar Labsmith sponsors Alex Rowan', 'reporter', 'x', False),
    ('Shipping today', 'alex_founder', 'x', True),
    ('Shipping today', 'alex_founder', 'reddit', False),
    ('Alex Rowan resigns', 'partner', 'x', False),
])
def test_identity_gate(text, author, source, expected):
    assert entity_extract.matches_person_identity(item(text, author, source),
        plan().person_identity_anchors, 'alex_founder') is expected


@pytest.mark.parametrize('hosted', [False, True])
def test_wrong_person_cannot_rank_even_with_high_scores_or_related_author(hosted):
    provider = Mock() if hosted else None
    wrong = candidate(item('Alex Rowan resigns as Alloy Works secretary', 'partner'))
    ranked = rerank.rerank_candidates(topic='Alex Rowan', plan=plan(), candidates=[wrong],
        provider=provider, model='fake' if hosted else None, shortlist_size=10,
        resolved_handles={'alex_founder', 'partner'}, primary_handle='alex_founder')
    assert ranked == []
    if hosted:
        provider.generate_json.assert_not_called()


def test_qualified_and_first_party_evidence_survives_fallback():
    for evidence in [item('Alex Rowan at Northstar Labs'), item('Shipping today', 'alex_founder')]:
        assert rerank.rerank_candidates(topic='Alex Rowan', plan=plan(),
            candidates=[candidate(evidence)], provider=None, model=None, shortlist_size=10,
            resolved_handles={'alex_founder'}, primary_handle='alex_founder')


def test_plan_roundtrip_and_validation():
    original = plan()
    assert schema.query_plan_from_dict(asdict(original)) == original
    raw = asdict(original)
    planner.validate_external_plan(raw)
    sanitized = planner._sanitize_plan(raw, topic='Alex Rowan', depth='default',
        available_sources=['x'], requested_sources=['x'])
    assert sanitized.person_identity_anchors == original.person_identity_anchors
    for invalid in ['Northstar Labs', None, [123], [''], ['!!!']]:
        raw['person_identity_anchors'] = invalid
        with pytest.raises(ValueError, match='person_identity_anchors'):
            planner.validate_external_plan(raw)


def test_namesake_does_not_expand_x_targets():
    bundle = schema.RetrievalBundle()
    bundle.items_by_source['x'] = [
        item('Alex Rowan resigns as Alloy Works secretary @alloynews'),
        item('Alex Rowan at Northstar Labs', 'qualified'),
    ]
    with patch.object(entity_extract, 'extract_entities', wraps=entity_extract.extract_entities) as extract:
        # Stop immediately after extraction; no network calls or real accounts.
        extract.side_effect = lambda reddit, posts, **kw: (_ for _ in ()).throw(StopIteration(posts))
        with pytest.raises(StopIteration) as exc:
            pipeline._run_supplemental_searches(topic='Alex Rowan', bundle=bundle, plan=plan(),
                config={}, depth='default', date_range=('2026-10-01', '2026-10-22'),
                runtime=schema.ProviderRuntime('local', '', '', x_search_backend='bird'),
                mock=False, rate_limited_sources=set(), rate_limit_lock=threading.Lock(),
                x_handle='alex_founder')
    assert [post['author_handle'] for post in exc.value.args[0]] == ['qualified']


def test_unscoped_plan_is_compatible():
    legacy_payload = asdict(plan())
    del legacy_payload['person_identity_anchors']
    planner.validate_external_plan(legacy_payload)
    legacy_plan = schema.query_plan_from_dict(legacy_payload)
    assert legacy_plan.person_identity_anchors == []
    evidence = item('Alex Rowan resigns as Alloy Works secretary')
    assert entity_extract.matches_person_identity(evidence, [])
    assert rerank.rerank_candidates(topic='Alex Rowan', plan=legacy_plan,
        candidates=[candidate(evidence)], provider=None, model=None, shortlist_size=10)
