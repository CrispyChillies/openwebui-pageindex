<script lang="ts">
	import { createEventDispatcher, getContext, onMount, tick } from 'svelte';
	import dayjs from '$lib/dayjs';
	import relativeTime from 'dayjs/plugin/relativeTime';

	dayjs.extend(relativeTime);

	import { toast } from 'svelte-sonner';

	import { getFiles, uploadFile } from '$lib/apis/files';
	import { getPageIndexStatus, pageIndexIndexFile } from '$lib/apis/retrieval';
	import { formatFileSize } from '$lib/utils';

	import { settings } from '$lib/stores';

	import MessageInput from './MessageInput.svelte';
	import FileItem from '../common/FileItem.svelte';
	import DocumentArrowUp from '../icons/DocumentArrowUp.svelte';

	const dispatch = createEventDispatcher();
	const i18n = getContext('i18n');

	export let createMessagePair: Function;
	export let stopResponse: Function;

	export let autoScroll = false;

	export let atSelectedModel;
	export let selectedModels = [''];

	export let history;

	export let prompt = '';
	export let files = [];
	export let messageInput = null;

	export let selectedToolIds = [];
	export let selectedFilterIds = [];

	export let showCommands = false;

	export let imageGenerationEnabled = false;
	export let codeInterpreterEnabled = false;
	export let webSearchEnabled = false;

	export let onUpload: Function = (e) => {};
	export let onChange = (e) => {};

	export let toolServers = [];

	export let dragged = false;

	let docsLoading = false;
	let uploading = false;
	let documents = [];
	let pageIndexStatuses = {};
	let uploadInputElement;

	const normalizeDocName = (doc) => doc?.filename ?? doc?.name ?? doc?.meta?.name ?? 'Document';
	const normalizeDocSize = (doc) => doc?.meta?.size ?? doc?.size ?? 0;

	const toAttachment = (doc) => ({
		...doc,
		id: doc?.id,
		name: normalizeDocName(doc),
		type: 'file',
		size: normalizeDocSize(doc),
		status: 'processed'
	});

	const isSelected = (docId) => files.some((file) => (file?.id ?? file?.file?.id) === docId);

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
				force_reindex: false
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

	const refreshDocuments = async () => {
		docsLoading = true;
		try {
			const res = await getFiles(localStorage.token);
			documents = (res ?? [])
				.filter((doc) => !!doc?.id)
				.sort((a, b) => (b?.updated_at ?? 0) - (a?.updated_at ?? 0));

			for (const doc of documents.slice(0, 30)) {
				loadPageIndexStatus(doc.id);
			}
		} catch (e) {
			toast.error($i18n.t('Failed to load documents'));
			documents = [];
		} finally {
			docsLoading = false;
		}
	};

	const toggleDocumentSelection = (doc) => {
		const docId = doc?.id;
		if (!docId) {
			return;
		}

		if (isSelected(docId)) {
			files = files.filter((file) => (file?.id ?? file?.file?.id) !== docId);
		} else {
			files = [...files, toAttachment(doc)];
			ensurePageIndexIndexing(docId);
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
					toggleDocumentSelection(uploaded);
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

	$: selectedDocuments = files.filter((file) => file?.type === 'file');

	onMount(async () => {
		await refreshDocuments();
	});
</script>

<div class="m-auto w-full max-w-6xl px-3 @2xl:px-16 py-10">
	<div class="mx-auto w-full max-w-5xl">
		<div class="flex items-center justify-between gap-3 mb-5">
			<div>
				<div class="text-2xl font-semibold text-gray-900 dark:text-gray-100">
					{$i18n.t('PageIndex Document Chat')}
				</div>
				<div class="text-sm text-gray-500 dark:text-gray-400 mt-1">
					{$i18n.t('Upload or select documents, then ask questions grounded in those files.')}
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

		{#if selectedDocuments.length > 0}
			<div class="mb-5">
				<div class="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2">
					{$i18n.t('Selected Documents')}
				</div>
				<div class="flex flex-wrap gap-2">
					{#each selectedDocuments as file, fileIdx (file?.id ?? fileIdx)}
						<div class="max-w-[320px]">
							<FileItem
								item={file}
								name={file?.name ?? normalizeDocName(file)}
								type="file"
								size={file?.size ?? normalizeDocSize(file)}
								small={true}
								dismissible={true}
								on:dismiss={() => {
									files = files.filter(
										(item) => (item?.id ?? item?.file?.id) !== (file?.id ?? file?.file?.id)
									);
								}}
							/>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<div
			class="rounded-2xl border border-gray-200/70 dark:border-gray-800/70 bg-white/40 dark:bg-gray-900/40 p-3 @md:p-4 mb-4"
		>
			<div class="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2">
				{$i18n.t('Your Documents')}
			</div>

			{#if docsLoading}
				<div class="text-sm text-gray-500 dark:text-gray-400 py-8 text-center">
					{$i18n.t('Loading documents...')}
				</div>
			{:else if documents.length === 0}
				<button
					type="button"
					class="w-full rounded-xl border border-dashed border-gray-300 dark:border-gray-700 py-8 px-4 text-center hover:border-gray-400 dark:hover:border-gray-600 transition"
					on:click={() => uploadInputElement?.click()}
				>
					<div class="text-sm font-medium text-gray-700 dark:text-gray-200">
						{$i18n.t('No documents yet')}
					</div>
					<div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
						{$i18n.t('Upload your first document to get started.')}
					</div>
				</button>
			{:else}
				<div class="grid grid-cols-1 @lg:grid-cols-2 gap-2.5 max-h-72 overflow-auto pr-0.5">
					{#each documents as doc (doc.id)}
						<button
							type="button"
							class="text-left rounded-xl border p-3 transition {isSelected(doc.id)
								? 'border-sky-400 bg-sky-50 dark:border-sky-700 dark:bg-sky-900/30'
								: 'border-gray-200 dark:border-gray-800 hover:border-gray-300 dark:hover:border-gray-700 bg-white/70 dark:bg-gray-900/50'}"
							on:click={() => toggleDocumentSelection(doc)}
						>
							<div class="flex items-start justify-between gap-2">
								<div class="font-medium text-sm line-clamp-1 text-gray-900 dark:text-gray-100">
									{normalizeDocName(doc)}
								</div>
								<div class="flex items-center gap-1.5">
									<div
										class="text-[11px] px-2 py-0.5 rounded-full {getStatusTone(
											pageIndexStatuses[doc.id]?.status ?? 'pending'
										)}"
									>
										{pageIndexStatuses[doc.id]?.status ?? 'pending'}
									</div>
									{#if isSelected(doc.id)}
										<div
											class="text-[11px] px-2 py-0.5 rounded-full bg-sky-600 text-white dark:bg-sky-500"
										>
											{$i18n.t('Selected')}
										</div>
									{/if}
								</div>
							</div>

							<div class="mt-2 text-xs text-gray-500 dark:text-gray-400 flex items-center gap-3">
								<span>{formatFileSize(normalizeDocSize(doc) || 0)}</span>
								{#if doc?.updated_at}
									<span>{dayjs(doc.updated_at * 1000).fromNow()}</span>
								{/if}
							</div>
						</button>
					{/each}
				</div>
			{/if}
		</div>

		<div class="text-base font-normal w-full pb-2">
			<MessageInput
				bind:this={messageInput}
				{history}
				{selectedModels}
				bind:files
				bind:prompt
				bind:autoScroll
				bind:selectedToolIds
				bind:selectedFilterIds
				bind:imageGenerationEnabled
				bind:codeInterpreterEnabled
				bind:webSearchEnabled
				bind:atSelectedModel
				bind:showCommands
				bind:dragged
				{toolServers}
				{stopResponse}
				{createMessagePair}
				placeholder={$i18n.t('Ask a question about your selected documents...')}
				{onChange}
				{onUpload}
				on:submit={(e) => {
					dispatch('submit', e.detail);
				}}
			/>
		</div>
	</div>
</div>
