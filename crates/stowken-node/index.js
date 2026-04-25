'use strict'

const { existsSync } = require('fs')
const { join } = require('path')

// napi-rs names the binary with a platform suffix when built with --platform.
// Try platform-specific name first, then fall back to plain name.
const platformSuffix = `${process.platform}-${process.arch}`
const binaryName = `stowken.${platformSuffix}.node`
const plainName = 'stowken.node'

let nativeBinding

const localPath = join(__dirname, binaryName)
const plainPath = join(__dirname, plainName)

if (existsSync(localPath)) {
  nativeBinding = require(localPath)
} else if (existsSync(plainPath)) {
  nativeBinding = require(plainPath)
} else {
  throw new Error(
    `Cannot find native module. Expected one of:\n  ${localPath}\n  ${plainPath}\n` +
    `Run 'npm run build' inside crates/stowken-node to build it.`
  )
}

module.exports = nativeBinding
