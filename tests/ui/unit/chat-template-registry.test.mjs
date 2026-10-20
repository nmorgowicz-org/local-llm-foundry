// Demoted from core/chat-template-updates.spec.js and core/spawn-wizard.spec.js: both only
// read static data and pure helpers from chat-template-registry.js.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as mod from '../../../static/js/features/chat-template-registry.js';

test('Qwen offers froggeric and Sharp, both tracking main', () => {
  const qwen = mod.getTemplatesForFamily('qwen');
  const registry = {
    names: qwen.map(tpl => tpl.name),
    defaultName: mod.getDefaultTemplateForFamily('qwen').name,
    pinned: qwen.filter(tpl => tpl.revision || tpl.transformed || tpl.version).map(tpl => tpl.name),
    sharp: mod.buildCommunityTemplateInstallRequest(mod.findTemplateByName('qwen-sharp')),
    qwenSuffix: mod.provenanceSuffix(qwen[0], 'qwen'),
    gemmaSuffix: mod.provenanceSuffix(mod.getTemplatesForFamily('gemma4')[0], 'gemma4'),
  };
  assert.deepEqual(registry.names, ['qwen-froggeric-fixed', 'qwen-sharp']);
  assert.equal(registry.defaultName, 'qwen-froggeric-fixed');
  assert.deepEqual(registry.pinned, []);
  assert.deepEqual(registry.sharp, {
    endpoint: '/api/chat-template/install-hf',
    body: { repo: 'peculiar-ragdoll/Qwen-Sharp-Chat-Templates', file: 'chat_template.jinja', name: 'qwen-sharp' },
  });
  // Two community templates are told apart by name; a mixed family still gets a label.
  assert.equal(registry.qwenSuffix, '');
  assert.equal(registry.gemmaSuffix, ' (Official)');
});

test('Community templates use the correct source for Qwen and Gemma 4', () => {
  const { COMMUNITY_TEMPLATES, buildCommunityTemplateInstallRequest } = mod;
  const templates = {
    templates: COMMUNITY_TEMPLATES,
    qwenInstall: buildCommunityTemplateInstallRequest(COMMUNITY_TEMPLATES.qwen[0]),
  };

  // Qwen offers froggeric's template (recommended, first) and the Sharp fork. Neither is
  // pinned to a version or commit: both track their repository's current `main`, and the
  // update checker offers new versions as upstream publishes them.
  const [froggeric, sharp] = templates.templates.qwen;
  assert.equal(froggeric.installEndpoint, '/api/chat-template/install-hf');
  assert.equal(froggeric.repo, 'froggeric/Qwen-Fixed-Chat-Templates');
  assert.equal(froggeric.revision, undefined);
  assert.equal(froggeric.version, undefined);
  assert.equal(templates.qwenInstall.body.revision, undefined);
  assert.equal(sharp.installEndpoint, '/api/chat-template/install-hf');
  assert.equal(sharp.repo, 'peculiar-ragdoll/Qwen-Sharp-Chat-Templates');
  assert.equal(sharp.revision, undefined);
  // Google's official template is the priority default (first entry);
  // jscott3201's agentic fork is kept as a fallback entry.
  assert.equal(templates.templates.gemma4[0].installEndpoint, '/api/chat-template/install-hf');
  assert.equal(templates.templates.gemma4[0].repo, 'google/gemma-4-31B-it');
  assert.equal(templates.templates.gemma4[1].installEndpoint, '/api/chat-template/install-url');
  assert.equal(
    templates.templates.gemma4[1].url,
    'https://raw.githubusercontent.com/jscott3201/llm-tuning/main/gemma4/chat_templates/custom_pub_chat_template_gemma4.jinja',
  );
});
