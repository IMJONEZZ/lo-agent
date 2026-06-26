from pathlib import Path

from local_harness.inference.capabilities import Capabilities
from local_harness.logits.grammar_stage import GrammarStage
from local_harness.logits.pipeline import LogitPipeline, StageStatus
from local_harness.logits.samplers import SamplerChain
from local_harness.skills.skill import SkillRegistry

SKILLS_DIR = Path(__file__).parent.parent / "skills"

LLAMACPP = Capabilities(server="llama.cpp", seed=True, logprobs=True, grammar="gbnf",
                        logit_bias=True, sampler_zoo={"min_p", "mirostat", "dry", "xtc"})
VLLM = Capabilities(server="vllm", seed=True, logprobs=True, grammar="guided",
                    logit_bias=True, sampler_zoo={"min_p", "top_k"}, parallel_n=True)
GENERIC = Capabilities()


def test_registry_loads_and_composes():
    reg = SkillRegistry(SKILLS_DIR)
    assert {"sql_core", "sql_select", "json_extract", "yes_no"} <= set(reg.names())
    sql = reg.get("sql_select")
    # imported rules from sql_core are merged in
    assert "condition" in sql.grammar.rules
    assert sql.grammar.validate("SELECT id, name FROM users WHERE age>=21;")
    assert not sql.grammar.validate("DROP TABLE users;")


def test_grammar_stage_per_backend():
    reg = SkillRegistry(SKILLS_DIR)
    sql, js = reg.get("sql_select"), reg.get("json_extract")

    res = GrammarStage(sql).compile_http(LLAMACPP)
    assert res.status == StageStatus.NATIVE and "grammar" in res.params
    res = GrammarStage(sql).compile_http(VLLM)
    assert res.status == StageStatus.NATIVE and "guided_grammar" in res.params
    res = GrammarStage(sql).compile_http(GENERIC)
    assert res.status == StageStatus.EMULATED

    res = GrammarStage(js).compile_http(LLAMACPP)
    assert res.params.get("json_schema", {}).get("type") == "object"
    res = GrammarStage(js).compile_http(VLLM)
    assert "guided_json" in res.params


def test_json_schema_validation():
    reg = SkillRegistry(SKILLS_DIR)
    js = reg.get("json_extract")
    assert js.validate_output('{"name": "Ada", "category": "person", "confidence": 0.9}')
    assert not js.validate_output('{"name": "Ada", "category": "verb", "confidence": 0.9}')
    assert not js.validate_output('{"name": "Ada"}')
    assert not js.validate_output("not json")


def test_sampler_chain_lowering():
    chain = SamplerChain({"min_p": 0.05, "mirostat": {"tau": 4.0}, "dry": {}, "nope": 1})
    res = chain.compile_http(LLAMACPP)
    assert res.status == StageStatus.NATIVE
    assert res.params["min_p"] == 0.05
    assert res.params["mirostat_tau"] == 4.0
    assert res.params["dry_multiplier"] == 0.8
    assert "dropped unsupported: ['nope']" in res.note

    res = chain.compile_http(VLLM)
    assert res.params == {"min_p": 0.05}  # only the vLLM-supported subset

    res = chain.compile_http(GENERIC)
    assert res.status == StageStatus.UNAVAILABLE


def test_pipeline_resolve_merges_and_reports():
    reg = SkillRegistry(SKILLS_DIR)
    pipeline = LogitPipeline([GrammarStage(reg.get("sql_select")), SamplerChain({"min_p": 0.1})])
    plan = pipeline.resolve(LLAMACPP)
    assert "grammar" in plan.body_params and plan.body_params["min_p"] == 0.1
    assert plan.status_of("grammar") == StageStatus.NATIVE
    d = plan.to_dict()
    assert {s["stage"] for s in d["stages"]} == {"grammar", "samplers"}
