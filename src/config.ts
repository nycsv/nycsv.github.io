import { defineCollection, z } from 'astro:content';

const noteSchema = z.object({
  title: z.string(),
  date: z.coerce.date(),
  tags: z.array(z.string()).optional(),
  description: z.string().optional(),
});

const posts = defineCollection({ type: 'content', schema: noteSchema });
const reviews = defineCollection({ type: 'content', schema: noteSchema });
const notes = defineCollection({ type: 'content', schema: noteSchema });

export const collections = { posts, reviews, notes };
