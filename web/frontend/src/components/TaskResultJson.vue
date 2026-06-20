<template>
  <details class="surface" style="margin-top: 16px">
    <summary style="cursor: pointer; font-weight: 500; color: #0f172a">TaskResult JSON</summary>
    <pre class="json-block" style="margin-top: 12px">{{ formatted }}</pre>
  </details>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ value: unknown }>()

const formatted = computed(() => JSON.stringify(props.value ?? {}, null, 2))
</script>
