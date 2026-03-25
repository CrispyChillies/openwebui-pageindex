<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import dayjs from '$lib/dayjs';
	import relativeTime from 'dayjs/plugin/relativeTime';
	import { toast } from 'svelte-sonner';

	import { getFiles, uploadFile } from '$lib/apis/files';
	import { getPageIndexStatus, pageIndexIndexFile } from '$lib/apis/retrieval';
	import { formatFileSize } from '$lib/utils';

	import DocumentArrowUp from '$lib/components/icons/DocumentArrowUp.svelte';

	dayjs.extend(relativeTime);

	const i18n = getContext('i18n');

	let docsLoading = false;
	let uploading = false;
	let documents = [];
	let pageIndexStatuses = {};
	let uploadInputElement;

	const normalizeDocName = (doc) => doc?.filename ?? doc?.name ?? doc?.meta?.name ?? 'Document';
	const normalizeDocSize = (doc) => doc?.meta?.size ?? doc?.size ?? 0;

	const getStatusTone = (status) => {
		switch (status) {
			case 'ready':
				return 'text-emerald-700 bg-emerald-50 dark:text-emerald-300 dark:bg-emerald-900/30';
			case 'processing':
				return 'text-amber-700 bg-amber-50 dark:text-amber-300 dark:bg-amber-900/30';
			case 'failed':
				return 'text-rose-700 bg-rose-50 dark:text-rose-300 dark:bg-rose-900/30';
			default:
				return 'text-gray-600 bg-gray-100 dark:text-gray-300 dark:bg-gray-800';
		}
	};

	const loadPageIndexStatus = async (fileId) => {
		if (!fileId) {
			return null;
		}

		try {
			const status = await getPageIndexStatus(localStorage.token, fileId);
			if (status) {
				pageIndexStatuses = {
					...pageIndexStatuses,
					[fileId]: status
				};
			}
			return status;
		} catch (e) {
			return null;
		}
	};

	const pollPageIndexStatus = async (fileId, attempt = 0) => {
		const status = await loadPageIndexStatus(fileId);
		if (!status) {
			if (attempt < 8) {
				setTimeout(() => {
					pollPageIndexStatus(fileId, attempt + 1);
				}, 1500);
			}
			return;
		}

		if (!['ready', 'failed'].includes(status.status) && attempt < 20) {
			setTimeout(() => {
				pollPageIndexStatus(fileId, attempt + 1);
			}, 1500);
		}
	};

	const ensurePageIndexIndexing = async (fileId) => {
		if (!fileId) {
			return;
		}

		const currentStatus = pageIndexStatuses[fileId]?.status;
		if (currentStatus === 'ready' || currentStatus === 'processing') {
			return;
		}

		try {
			pageIndexStatuses = {
				...pageIndexStatuses,
				[fileId]: {
					file_id: fileId,
					status: 'processing'
				}
			};

			await pageIndexIndexFile(localStorage.token, {
				file_id: fileId,
				force_reindex: false,
				index_options: {
					if_add_doc_description: 'yes'
				}
			});

			pollPageIndexStatus(fileId, 0);
		} catch (e) {
			pageIndexStatuses = {
				...pageIndexStatuses,
				[fileId]: {
					file_id: fileId,
					status: 'failed',
					error_message: String(e)
				}
			};
		}
	};

	const refreshDocuments = async () => {
		docsLoading = true;
		try {
			const res = await getFiles(localStorage.token);
			documents = (res ?? [])
				.filter((doc) => !!doc?.id)
				.sort((a, b) => (b?.updated_at ?? 0) - (a?.updated_at ?? 0));

			for (const doc of documents.slice(0, 100)) {
				loadPageIndexStatus(doc.id);
			}
		} catch (e) {
			toast.error($i18n.t('Failed to load documents'));
			documents = [];
		} finally {
			docsLoading = false;
		}
	};

	const uploadDocuments = async (event) => {
		const inputFiles = Array.from(event?.target?.files ?? []);
		if (inputFiles.length === 0) {
			return;
		}

		uploading = true;
		for (const file of inputFiles) {
			try {
				const uploaded = await uploadFile(localStorage.token, file, null, true);
				if (uploaded?.id) {
					documents = [uploaded, ...documents.filter((doc) => doc?.id !== uploaded.id)];
					await ensurePageIndexIndexing(uploaded.id);
					pollPageIndexStatus(uploaded.id, 0);
				}
			} catch (e) {
				toast.error(
					$i18n.t('Error uploading file: {{error}}', {
						error: e?.message ?? e ?? 'Unknown error'
					})
				);
			}
		}

		uploading = false;
		if (uploadInputElement) {
			uploadInputElement.value = '';
		}
	};

	onMount(async () => {
		await refreshDocuments();
	});
</script>

<div class="w-full max-w-6xl mx-auto px-4 @2xl:px-16 py-8">
	<div class="flex items-center justify-between gap-3 mb-5">
		<div>
			<div class="text-2xl font-semibold text-gray-900 dark:text-gray-100">
				{$i18n.t('Documents')}
			</div>
			<div class="text-sm text-gray-500 dark:text-gray-400 mt-1">
				{$i18n.t('View and upload all your document files.')}
			</div>
		</div>

		<div class="shrink-0">
			<input
				bind:this={uploadInputElement}
				type="file"
				multiple
				class="hidden"
				on:change={uploadDocuments}
			/>

			<button
				type="button"
				class="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-white transition"
				on:click={() => uploadInputElement?.click()}
				disabled={uploading}
			>
				<DocumentArrowUp className="size-4" />
				{uploading ? $i18n.t('Uploading...') : $i18n.t('Upload Documents')}
			</button>
		</div>
	</div>

	<div
		class="rounded-2xl border border-gray-200/70 dark:border-gray-800/70 bg-white/40 dark:bg-gray-900/40 p-3 @md:p-4"
	>
		{#if docsLoading}
			<div class="text-sm text-gray-500 dark:text-gray-400 py-8 text-center">
				{$i18n.t('Loading documents...')}
			</div>
		{:else if documents.length === 0}
			<div class="text-sm text-gray-500 dark:text-gray-400 py-8 text-center">
				{$i18n.t('No documents uploaded yet.')}
			</div>
		{:else}
			<div class="space-y-2.5 max-h-[72vh] overflow-auto pr-1">
				{#each documents as doc (doc.id)}
					<div
						class="rounded-xl border p-3 border-gray-200 dark:border-gray-800 bg-white/70 dark:bg-gray-900/50"
					>
						<div class="flex items-start justify-between gap-2">
							<div class="font-medium text-sm line-clamp-1 text-gray-900 dark:text-gray-100">
								{normalizeDocName(doc)}
							</div>
							<div class="flex items-center gap-2">
								<div
									class="text-[11px] px-2 py-0.5 rounded-full {getStatusTone(
										pageIndexStatuses[doc.id]?.status ?? 'pending'
									)}"
								>
									{pageIndexStatuses[doc.id]?.status ?? 'pending'}
								</div>
								{#if (pageIndexStatuses[doc.id]?.status ?? 'pending') !== 'ready'}
									<button
										type="button"
										class="text-xs px-2.5 py-1 rounded-lg bg-sky-600 text-white hover:bg-sky-500 transition"
										on:click={() => ensurePageIndexIndexing(doc.id)}
									>
										{$i18n.t('Index')}
									</button>
								{/if}
							</div>
						</div>

						<div class="mt-2 text-xs text-gray-500 dark:text-gray-400 flex items-center gap-3">
							<span>{formatFileSize(normalizeDocSize(doc) || 0)}</span>
							{#if doc?.updated_at}
								<span>{dayjs(doc.updated_at * 1000).fromNow()}</span>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>
