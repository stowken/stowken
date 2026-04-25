'use strict';
/**
 * Stowken Node.js binding tests.
 *
 * Plain Node.js — no test framework required.
 *
 * Run:
 *   node test.js
 *
 * Build the native module first if needed:
 *   npm run build   (from this directory)
 */

const assert = require('assert');
const { Stowken } = require('./index');

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeConv(id, system = 'You are helpful.', user = 'Hello', assistant = 'Hi!') {
  return {
    id,
    model: 'gpt-4',
    tokenizer: 'cl100k_base',
    messages: [
      { role: 'system',    content: system },
      { role: 'user',      content: user },
      { role: 'assistant', content: assistant },
    ],
  };
}

function makeTokenConv(id, sysTokens, userTokens) {
  return {
    id,
    model: 'gpt-4',
    tokenizer: 'cl100k_base',
    messages: [
      { role: 'system', tokens: sysTokens },
      { role: 'user',   tokens: userTokens },
    ],
  };
}

let passed = 0;
let failed = 0;

async function test(name, fn) {
  try {
    await fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error(`      ${err.message}`);
    failed++;
  }
}

// ── StoreResult shape ─────────────────────────────────────────────────────────

async function testStoreResult() {
  console.log('\nStoreResult shape');
  const vault = new Stowken();

  await test('has id field matching input id', async () => {
    const r = await vault.store(makeConv('sr-1'));
    assert.strictEqual(r.id, 'sr-1');
  });

  await test('has numeric newSegments > 0 on first store', async () => {
    const v = new Stowken();
    const r = await v.store(makeConv('sr-2'));
    assert.ok(typeof r.newSegments === 'number');
    assert.ok(r.newSegments > 0);
  });

  await test('totalSegments equals newSegments + dedupedSegments', async () => {
    const v = new Stowken();
    const r = await v.store(makeConv('sr-3'));
    assert.strictEqual(r.totalSegments, r.newSegments + r.dedupedSegments);
  });

  await test('compressionRatio is a number', async () => {
    const v = new Stowken();
    const r = await v.store(makeConv('sr-4'));
    assert.ok(typeof r.compressionRatio === 'number');
    assert.ok(Number.isFinite(r.compressionRatio));
  });
}

// ── Store / retrieve roundtrip ─────────────────────────────────────────────────

async function testStoreRetrieve() {
  console.log('\nStore/retrieve roundtrip');
  const vault = new Stowken();

  await test('retrieve returns object with expected keys', async () => {
    await vault.store(makeConv('rt-1'));
    const r = await vault.retrieve('rt-1');
    assert.strictEqual(r.id, 'rt-1');
    assert.strictEqual(r.model, 'gpt-4');
    assert.ok(Array.isArray(r.segments));
  });

  await test('segments have expected fields', async () => {
    await vault.store(makeConv('rt-2'));
    const r = await vault.retrieve('rt-2');
    for (const seg of r.segments) {
      assert.ok(typeof seg.segmentType === 'string');
      assert.ok(typeof seg.tokenCount === 'number');
      assert.ok(typeof seg.hash === 'string');
      assert.ok(seg.tokenCount > 0);
    }
  });

  await test('pre-tokenized tokens stored and reported correctly', async () => {
    const sysTokens = Array.from({ length: 50 }, (_, i) => i);
    const userTokens = [10, 20, 30];
    await vault.store(makeTokenConv('rt-tok', sysTokens, userTokens));
    const r = await vault.retrieve('rt-tok');
    assert.strictEqual(r.totalTokens, 53);
  });

  await test('retrieve unknown id rejects', async () => {
    await assert.rejects(() => vault.retrieve('no-such-id'));
  });
}

// ── Deduplication ──────────────────────────────────────────────────────────────

async function testDeduplication() {
  console.log('\nDeduplication');
  const vault = new Stowken();

  await test('second identical store reports dedupedSegments > 0', async () => {
    await vault.store(makeConv('dup-1'));
    const r2 = await vault.store(makeConv('dup-2'));
    assert.ok(r2.dedupedSegments > 0, `expected dedup, got ${JSON.stringify(r2)}`);
  });

  await test('bytesSaved positive on exact dup', async () => {
    await vault.store(makeConv('bs-1'));
    const r2 = await vault.store(makeConv('bs-2'));
    assert.ok(r2.bytesSaved > 0);
  });

  await test('uniqueSegments does not grow on exact dup', async () => {
    const v = new Stowken();
    await v.store(makeConv('us-1'));
    const before = (await v.stats()).uniqueSegments;
    await v.store(makeConv('us-2'));
    const after = (await v.stats()).uniqueSegments;
    assert.strictEqual(after, before);
  });
}

// ── Stats ──────────────────────────────────────────────────────────────────────

async function testStats() {
  console.log('\nStats');

  await test('fresh vault has zero conversations', async () => {
    const v = new Stowken();
    const s = await v.stats();
    assert.strictEqual(s.totalConversations, 0);
  });

  await test('stats after store has correct fields', async () => {
    const v = new Stowken();
    await v.store(makeConv('st-1'));
    const s = await v.stats();
    assert.ok(s.totalConversations >= 1);
    assert.ok(s.totalTokens > 0);
    assert.ok(s.uniqueSegments >= 1);
    assert.ok(typeof s.compressionRatio === 'number');
    assert.ok(typeof s.savingsPercentage === 'number');
  });
}

// ── Delete ─────────────────────────────────────────────────────────────────────

async function testDelete() {
  console.log('\nDelete');
  const vault = new Stowken();

  await test('delete removes conversation', async () => {
    await vault.store(makeConv('del-1'));
    await vault.delete('del-1');
    await assert.rejects(() => vault.retrieve('del-1'));
  });

  await test('delete nonexistent id rejects', async () => {
    await assert.rejects(() => vault.delete('does-not-exist'));
  });
}

// ── GC ─────────────────────────────────────────────────────────────────────────

async function testGC() {
  console.log('\nGC');
  const vault = new Stowken();

  await test('gc returns a number', async () => {
    const n = await vault.gc();
    assert.ok(typeof n === 'number');
  });

  await test('gc frees segments after delete', async () => {
    const v = new Stowken();
    await v.store(makeConv('gc-1'));
    await v.delete('gc-1');
    const freed = await v.gc();
    assert.ok(freed > 0, `expected freed > 0, got ${freed}`);
  });
}

// ── Segment stats ──────────────────────────────────────────────────────────────

async function testSegmentStats() {
  console.log('\nSegment stats');
  const vault = new Stowken();

  await test('returns array of objects with expected fields', async () => {
    await vault.store(makeConv('ss-1'));
    const stats = await vault.segmentStats();
    assert.ok(Array.isArray(stats));
    assert.ok(stats.length > 0);
    for (const s of stats) {
      assert.ok(typeof s.segmentType === 'string');
      assert.ok(typeof s.uniqueCount === 'number');
      assert.ok(typeof s.totalReferences === 'number');
      assert.ok(typeof s.dedupRatio === 'number');
      assert.ok(typeof s.totalTokens === 'number');
      assert.ok(typeof s.compressedBytes === 'number');
    }
  });

  await test('system prompt segment is present', async () => {
    await vault.store(makeConv('ss-2'));
    const stats = await vault.segmentStats();
    const types = stats.map(s => s.segmentType);
    assert.ok(types.some(t => t.toLowerCase().includes('system')), `expected system segment, got ${types}`);
  });
}

// ── Compression disabled ───────────────────────────────────────────────────────

async function testCompressionFlag() {
  console.log('\nCompression flag');

  await test('vault with compression disabled still stores', async () => {
    const v = new Stowken(false);
    const r = await v.store(makeConv('nc-1'));
    assert.ok(r.newSegments > 0);
  });
}

// ── Runner ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log('Stowken Node.js binding tests');
  console.log('=================================');

  await testStoreResult();
  await testStoreRetrieve();
  await testDeduplication();
  await testStats();
  await testDelete();
  await testGC();
  await testSegmentStats();
  await testCompressionFlag();

  console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

main().catch(err => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
