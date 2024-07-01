/**
 * @file @nuogz/common-eslint-config
 * @author DanoR
 * @version 5.2.0 2024.06.18 15
 * @requires globals
 * @requires @eslint/js
 * @requires @stylistic/eslint-plugin-js
 * @requires eslint-plugin-vue (optional)
 */


import { readFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

import globals from 'globals';
import js from '@eslint/js';
import stylistic from '@stylistic/eslint-plugin-js';



const PKG = JSON.parse(readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), 'package.json'), 'utf8'));

/** @type {Set<string>} */
const typesSource = new Set(PKG.typesSource instanceof Array ? PKG.typesSource : []);


/** @type {import('eslint').Linter.FlatConfig[]} */
const configs = [
	{
		name: 'ignore-dist',
		ignores: ['dist/**'],
	},
	{
		name: 'rule-base',
		plugins: { stylistic },
		rules: {
			...js.configs.recommended.rules,
			...stylistic.configs['disable-legacy'].rules,

			stylistic$indent: [2, 'tab', { ignoredNodes: ['TemplateLiteral', 'CallExpression>ObjectExpression:not(:first-child)'], ignoreComments: true, SwitchCase: 1 }],
			stylistic$linebreakStyle: [2, 'unix'],
			stylistic$quotes: [2, 'single', { avoidEscape: true, allowTemplateLiterals: true }],
			stylistic$commaDangle: [2, 'only-multiline'],
			semi: [2],
			noUnusedVars: [2, { vars: 'all', args: 'none' }],
			noVar: [2],
			noConsole: [2],
			noShadow: [2, { ignoreOnInitialization: true }],
			noConstantBinaryExpression: [0],
			requireAtomicUpdates: [2, { allowProperties: true }],
		},
	},
];



if(typesSource.has('node') && typesSource.has('browser')) {
	configs.push({
		name: 'globals-node-with-browser',
		ignores: [
			'**/*.pure.?(c|m)js',
			'src/**/*.?(c|m)js',
			'!src/**/*.{api,lib,map}.?(c|m)js',
			'!src/**/*.lib/**/*.?(c|m)js'
		],
		languageOptions: { globals: globals.node },
	});

	configs.push({
		name: 'globals-browser-with-node',
		files: ['src/**/*.?(c|m)js'],
		ignores: [
			'eslint.config.?(c|m)js',
			'**/*.pure.?(c|m)js',
			'src/**/*.{api,lib,map}.?(c|m)js',
			'src/**/*.lib/**/*.?(c|m)js'
		],
		languageOptions: { globals: globals.browser },
	});
}
else if(typesSource.has('node')) {
	configs.push({
		name: 'globals-node-only',
		languageOptions: { globals: globals.node }
	});
}
else if(typesSource.has('browser')) {
	configs.push({
		name: 'globals-browser-only',
		ignores: ['eslint.config.?(c|m)js'],
		languageOptions: { globals: globals.browser },
	});

	configs.push({
		name: 'globals-node-config-patch',
		files: ['eslint.config.?(c|m)js'],
		languageOptions: { globals: globals.node },
	});
}


if(typesSource.has('vue')) {
	const vue = (await import('eslint-plugin-vue')).default;

	const [, configVueBase, configVueEssential, configVueRecommendedStrongly, configVueRecommended] = vue.configs['flat/recommended'];

	configs.push({
		name: 'rule-vue',
		files: ['**/*.vue'],
		plugins: configVueBase.plugins,
		languageOptions: configVueBase.languageOptions,
		processor: configVueBase.processor,
		rules: {
			...configVueBase.rules,
			...configVueEssential.rules,
			...configVueRecommendedStrongly.rules,
			...configVueRecommended.rules,

			stylistic$indent: [0],
			vue$htmlIndent: [2, 'tab'],
			vue$scriptIndent: [2, 'tab', { baseIndent: 0 }],
			vue$htmlSelfClosing: [1, { html: { void: 'always' } }],
			vue$maxAttributesPerLine: [0],
			vue$mustacheInterpolationSpacing: [0],
			vue$singlelineHtmlElementContentNewline: [0],
			vue$noVHtml: [0],
			vue$firstAttributeLinebreak: [0],
			vue$htmlClosingBracketNewline: [0],
			vue$multiWordComponentNames: [0],
		},
	});
}


const typesNodeConfig = [...typesSource.values()].filter(typeSource => typeSource.endsWith('@node-config'));
if(typesNodeConfig.length) {
	const configBrowserOnly = configs.find(config => config.name == 'globals-browser-only');
	const configBrowserWithNode = configs.find(config => config.name == 'globals-browser-with-node');

	let configNodeConfig = configs.find(config => config.name == 'globals-node-config-patch');
	if(!configNodeConfig) {
		configs.push(configNodeConfig = {
			name: 'globals-node-config-patch',
			files: [],
			languageOptions: { globals: globals.node },
		});
	}

	for(const typeNodeConfig of typesNodeConfig) {
		const [typePackage] = typeNodeConfig.split('@');

		configNodeConfig.files.push(`**/${typePackage}.config.?(c|m)js`);

		configBrowserOnly?.ignores.push(`**/${typePackage}.config.?(c|m)js`);
		configBrowserWithNode?.ignores.push(`**/${typePackage}.config.?(c|m)js`);
	}
}



for(const config of configs) {
	const rules = config.rules;
	if(typeof rules != 'object') { continue; }

	for(const key of Object.keys(rules)) {
		const [plugin, keyCamel] = key.includes('$') ? key.split('$') : [null, key];
		const keyKebab = `${plugin ? `${plugin}/` : ''}${keyCamel.split(/(?=[A-Z])/).join('-').toLowerCase()}`;

		if(keyKebab != key) {
			rules[keyKebab] = rules[key];

			delete rules[key];
		}
	}
}



export default configs;