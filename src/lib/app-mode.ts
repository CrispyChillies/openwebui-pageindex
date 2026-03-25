import { derived } from 'svelte/store';
import { config } from '$lib/stores';

export type OpenWebUIAppMode = 'default' | 'pageindex';

export const normalizeAppMode = (value?: string | null): OpenWebUIAppMode => {
	return value === 'pageindex' ? 'pageindex' : 'default';
};

export const appMode = derived(config, ($config) => normalizeAppMode($config?.app_mode));

export const is_default_mode = derived(appMode, ($appMode) => $appMode === 'default');
export const is_pageindex_mode = derived(appMode, ($appMode) => $appMode === 'pageindex');
